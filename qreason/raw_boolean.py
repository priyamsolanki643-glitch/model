"""Raw Boolean-clause candidate selection without verifier-feature leakage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .boolean_tasks import _clause_values
from .core import QReasonConfig, QReasonCore
from .synthetic import top1_accuracy
from .trainable import parameter_count


@dataclass(frozen=True)
class RawBooleanTask:
    candidates: Tensor
    clauses: Tensor
    labels: Tensor

    @property
    def candidate_features(self) -> Tensor:
        batch, candidate_count, variable_count = self.candidates.shape
        clause_features = self.clauses.float().reshape(batch, -1)
        return torch.cat(
            [
                self.candidates,
                clause_features.unsqueeze(1).expand(-1, candidate_count, -1),
            ],
            dim=-1,
        )


def make_raw_boolean_task(
    *,
    samples: int = 512,
    candidate_count: int = 8,
    variable_count: int = 6,
    clause_count: int = 8,
    seed: int = 101,
) -> RawBooleanTask:
    if min(samples, candidate_count, variable_count, clause_count) < 1:
        raise ValueError("all task dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.randint(
        0, 2, (samples, candidate_count, variable_count), generator=generator
    ).float()
    clauses = torch.randint(
        1, variable_count + 1, (samples, clause_count, 3), generator=generator
    )
    signs = torch.randint(
        0, 2, (samples, clause_count, 3), generator=generator
    ).bool()
    clauses = torch.where(signs, clauses, -clauses)
    valid = torch.stack(
        [
            _clause_values(candidates[index], clauses[index]).min(dim=-1).values
            for index in range(samples)
        ]
    )
    labels = valid.argmax(dim=-1)
    return RawBooleanTask(candidates, clauses, labels)


class RawQReasonClassifier(nn.Module):
    def __init__(self, feature_dim: int, config: QReasonConfig) -> None:
        super().__init__()
        self.candidate_count = config.state_dim
        self.amplitude = nn.Linear(feature_dim, 2)
        self.quality = nn.Linear(feature_dim, 1)
        self.core = QReasonCore(config)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.amplitude(features)
        amplitudes = torch.complex(raw[..., 0], raw[..., 1])
        quality = torch.tanh(self.quality(features)).squeeze(-1)
        return self.core(amplitudes, quality).scores


class RawRealClassifier(nn.Module):
    def __init__(self, feature_dim: int, candidate_count: int, hidden_dim: int = 5) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.scorer(features).squeeze(-1)


def train_raw_model(
    model: nn.Module,
    task: RawBooleanTask,
    *,
    steps: int = 250,
    learning_rate: float = 2e-3,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    features = task.candidate_features
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def evaluate_raw(model: nn.Module, task: RawBooleanTask) -> dict[str, float | int]:
    scores = model(task.candidate_features)
    return {
        "top1_accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(nn.functional.cross_entropy(scores, task.labels)),
        "finite": float(torch.isfinite(scores).all()),
        "parameters": parameter_count(model),
    }
