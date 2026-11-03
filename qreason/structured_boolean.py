"""Structured candidate-clause encoding and fair reasoning controls."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

import torch
from torch import Tensor, nn

from .core import QReasonConfig, QReasonCore
from .synthetic import top1_accuracy
from .trainable import parameter_count
from .unique_boolean import UniqueBooleanTask, make_unique_boolean_task, verify_unique


@dataclass(frozen=True)
class StructuredBooleanTask:
    features: Tensor
    labels: Tensor


def encode_candidate_clause_interactions(task: UniqueBooleanTask) -> StructuredBooleanTask:
    """Encode literal satisfaction per candidate without exposing labels.

    The encoder performs the same deterministic clause evaluation a symbolic
    solver would use, but returns only differentiable interaction features to
    every model. This isolates the reasoning core from raw integer parsing.
    """
    batch, candidate_count, variable_count = task.candidates.shape
    clause_count = task.clauses.shape[1]
    literals = task.candidates.unsqueeze(2).expand(-1, -1, clause_count, -1)
    variable_ids = task.clauses.abs() - 1
    selected = literals.gather(
        -1,
        variable_ids.unsqueeze(1).expand(-1, candidate_count, -1, -1),
    )
    signs = (task.clauses >= 0).float().unsqueeze(1)
    signed = selected * signs + (1.0 - selected) * (1.0 - signs)
    clause_satisfaction = signed.max(dim=-1).values
    clause_margin = clause_satisfaction.mean(dim=-1, keepdim=True)
    candidate_bits = task.candidates
    features = torch.cat([candidate_bits, signed.reshape(
        batch, candidate_count, clause_count * 3
    ), clause_margin], dim=-1)
    return StructuredBooleanTask(features, task.labels)


class SharedFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, width), nn.Tanh())

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class StructuredRealControl(nn.Module):
    def __init__(self, feature_dim: int, candidate_count: int, width: int = 8) -> None:
        super().__init__()
        self.encoder = SharedFeatureEncoder(feature_dim, width)
        self.score = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.score(self.encoder(features)).squeeze(-1)


class StructuredRecurrentControl(nn.Module):
    def __init__(self, feature_dim: int, candidate_count: int, width: int = 8) -> None:
        super().__init__()
        self.encoder = SharedFeatureEncoder(feature_dim, width)
        self.recurrent = nn.GRUCell(width, width)
        self.score = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> Tensor:
        hidden = torch.zeros(
            features.shape[0], features.shape[-1] if False else self.recurrent.hidden_size,
            device=features.device, dtype=features.dtype
        )
        encoded = self.encoder(features)
        hidden = self.recurrent(encoded.reshape(-1, encoded.shape[-1]), hidden.repeat_interleave(
            encoded.shape[1], dim=0
        ))
        return self.score(hidden).reshape(features.shape[0], features.shape[1])


class StructuredQReason(nn.Module):
    def __init__(self, feature_dim: int, config: QReasonConfig, width: int = 8) -> None:
        super().__init__()
        self.encoder = SharedFeatureEncoder(feature_dim, width)
        self.amplitude = nn.Linear(width, 2)
        self.quality = nn.Linear(width, 1)
        self.core = QReasonCore(config)

    def forward(self, features: Tensor) -> Tensor:
        encoded = self.encoder(features)
        raw = self.amplitude(encoded)
        amplitudes = torch.complex(raw[..., 0], raw[..., 1])
        quality = torch.tanh(self.quality(encoded)).squeeze(-1)
        return self.core(amplitudes, quality).scores


def train_structured(model: nn.Module, task: StructuredBooleanTask, steps: int = 250) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(task.features), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate_structured(model: nn.Module, task: StructuredBooleanTask) -> dict[str, float]:
    scores = model(task.features)
    return {
        "accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(nn.functional.cross_entropy(scores, task.labels)),
        "finite": float(torch.isfinite(scores).all()),
    }


def run_structured_comparison(seeds: tuple[int, ...] = (3, 5, 7, 11, 13)) -> dict[str, object]:
    results: dict[str, list[float]] = {
        "real": [], "recurrent": [], "qreason": [],
    }
    losses: dict[str, list[float]] = {
        "real": [], "recurrent": [], "qreason": [],
    }
    parameter_counts: dict[str, int] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        train = encode_candidate_clause_interactions(make_unique_boolean_task(samples=1024, seed=701))
        test = encode_candidate_clause_interactions(make_unique_boolean_task(samples=512, seed=801))
        feature_dim = train.features.shape[-1]
        config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
        models = {
            "real": StructuredRealControl(feature_dim, 8),
            "recurrent": StructuredRecurrentControl(feature_dim, 8),
            "qreason": StructuredQReason(feature_dim, config),
        }
        for name, model in models.items():
            train_structured(model, train)
            metrics = evaluate_structured(model, test)
            results[name].append(metrics["accuracy"])
            losses[name].append(metrics["cross_entropy"])
            parameter_counts[name] = parameter_count(model)
    return {
        name: {
            "accuracy": {"mean": mean(values), "std": stdev(values)},
            "cross_entropy": {"mean": mean(losses[name]), "std": stdev(losses[name])},
            "parameters": parameter_counts[name],
        }
        for name, values in results.items()
    } | {"seeds": len(seeds)}
