"""Boolean constraint candidate-selection benchmark for QReason."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .core import QReasonConfig, QReasonCore
from .synthetic import top1_accuracy


@dataclass(frozen=True)
class BooleanTask:
    candidates: Tensor
    labels: Tensor


def _clause_values(
    assignments: Tensor,
    clauses: Tensor,
) -> Tensor:
    """Return one Boolean satisfaction value per assignment and clause."""
    variables = assignments[..., clauses.abs() - 1]
    signs = torch.where(clauses >= 0, 1.0, -1.0)
    literals = torch.where(signs > 0, variables, 1.0 - variables)
    return literals.max(dim=-1).values


def _satisfies(assignments: Tensor, clauses: Tensor) -> Tensor:
    return _clause_values(assignments, clauses).min(dim=-1).values


def make_boolean_task(
    *,
    samples: int = 512,
    candidate_count: int = 8,
    variable_count: int = 6,
    clause_count: int = 8,
    seed: int = 101,
) -> BooleanTask:
    """Create candidate assignments with one satisfying assignment per sample.

    Candidate tensors concatenate assignment bits with per-clause truth values.
    Labels are retained only for scoring; no verifier label is supplied to the
    model during inference or training.
    """
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
    satisfaction = torch.stack([
        _clause_values(candidates[index], clauses[index])
        for index in range(samples)
    ])
    valid = satisfaction.min(dim=-1).values
    valid_count = valid.sum(dim=-1)
    needs_repair = valid_count == 0
    if needs_repair.any():
        candidates[needs_repair, 0] = 0.0
        satisfaction[needs_repair] = torch.stack([
            _clause_values(candidates[index], clauses[index])
            for index in torch.where(needs_repair)[0]
        ])
        valid = satisfaction.min(dim=-1).values
    labels = valid.argmax(dim=-1)
    # Encode the rules in every candidate row through clause satisfaction.
    features = torch.cat([candidates, satisfaction], dim=-1)
    return BooleanTask(features, labels)


class CandidateQReasonClassifier(nn.Module):
    """Scores candidate rows through shared projection and QReason dynamics."""

    def __init__(self, feature_dim: int, config: QReasonConfig) -> None:
        super().__init__()
        self.candidate_count = config.state_dim
        self.amplitude = nn.Linear(feature_dim, 2)
        self.quality = nn.Linear(feature_dim, 1)
        self.core = QReasonCore(config)

    def forward(self, candidates: Tensor) -> Tensor:
        if candidates.ndim != 3 or candidates.shape[1] != self.candidate_count:
            raise ValueError("candidates must have shape [batch, candidate_count, features]")
        raw = self.amplitude(candidates)
        amplitudes = torch.complex(raw[..., 0], raw[..., 1])
        quality = torch.tanh(self.quality(candidates)).squeeze(-1)
        return self.core(amplitudes, quality).scores


class CandidateRealClassifier(nn.Module):
    """Classical candidate scorer with the same shared input interface."""

    def __init__(self, feature_dim: int, candidate_count: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, candidates: Tensor) -> Tensor:
        return self.scorer(candidates).squeeze(-1)


def evaluate_boolean(model: nn.Module, task: BooleanTask) -> dict[str, float]:
    with torch.no_grad():
        scores = model(task.candidates)
        loss = nn.functional.cross_entropy(scores, task.labels)
    return {
        "top1_accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(loss),
        "finite": float(torch.isfinite(scores).all()),
    }


def train_boolean_model(
    model: nn.Module,
    task: BooleanTask,
    *,
    steps: int = 250,
    learning_rate: float = 2e-3,
) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(task.candidates), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
