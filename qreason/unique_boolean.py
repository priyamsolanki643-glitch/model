"""Unique-solution Boolean constraint benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .boolean_tasks import _clause_values
from .raw_boolean import RawQReasonClassifier, RawRealClassifier
from .synthetic import top1_accuracy
from .trainable import parameter_count


@dataclass(frozen=True)
class UniqueBooleanTask:
    candidates: Tensor
    clauses: Tensor
    labels: Tensor

    @property
    def candidate_features(self) -> Tensor:
        batch, candidate_count, _ = self.candidates.shape
        clause_features = self.clauses.float().reshape(batch, -1)
        return torch.cat(
            [
                self.candidates,
                clause_features.unsqueeze(1).expand(-1, candidate_count, -1),
            ],
            dim=-1,
        )


def _literal_for_target(target: Tensor, variable: int) -> int:
    value = int(target[variable])
    return variable + 1 if value else -(variable + 1)


def make_unique_boolean_task(
    *,
    samples: int = 512,
    candidate_count: int = 8,
    variable_count: int = 6,
    seed: int = 501,
) -> UniqueBooleanTask:
    if min(samples, candidate_count, variable_count) < 1:
        raise ValueError("all task dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.randint(
        0, 2, (samples, candidate_count, variable_count), generator=generator
    ).float()
    labels = torch.randint(candidate_count, (samples,), generator=generator)
    clauses = torch.empty(samples, candidate_count - 1, 3, dtype=torch.long)

    for sample in range(samples):
        target_index = int(labels[sample])
        target = candidates[sample, target_index]
        cursor = 0
        for candidate_index in range(candidate_count):
            if candidate_index == target_index:
                continue
            differences = torch.where(candidates[sample, candidate_index] != target)[0]
            if differences.numel() == 0:
                replacement = (int(target_index) + 1) % candidate_count
                candidates[sample, candidate_index] = 1.0 - target
                differences = torch.where(candidates[sample, candidate_index] != target)[0]
                _ = replacement
            chosen = differences[
                torch.randint(differences.numel(), (3,), generator=generator)
            ]
            clauses[sample, cursor] = torch.tensor([
                _literal_for_target(target, int(variable)) for variable in chosen
            ])
            cursor += 1

    return UniqueBooleanTask(candidates, clauses, labels)


@torch.no_grad()
def verify_unique(task: UniqueBooleanTask) -> bool:
    valid_counts = []
    for sample in range(task.candidates.shape[0]):
        valid = _clause_values(
            task.candidates[sample], task.clauses[sample]
        ).min(dim=-1).values
        valid_counts.append(int(valid.sum()))
    return all(count == 1 for count in valid_counts)


def train_unique_model(
    model: torch.nn.Module,
    task: UniqueBooleanTask,
    *,
    steps: int = 250,
    learning_rate: float = 2e-3,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    features = task.candidate_features
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(features), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def evaluate_unique(model: torch.nn.Module, task: UniqueBooleanTask) -> dict[str, float | int]:
    scores = model(task.candidate_features)
    return {
        "top1_accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(torch.nn.functional.cross_entropy(scores, task.labels)),
        "finite": float(torch.isfinite(scores).all()),
        "parameters": parameter_count(model),
    }


__all__ = [
    "RawQReasonClassifier",
    "RawRealClassifier",
    "UniqueBooleanTask",
    "evaluate_unique",
    "make_unique_boolean_task",
    "train_unique_model",
    "verify_unique",
]
