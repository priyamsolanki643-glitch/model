"""Verifier-free synthetic task generation and evaluation.

Labels are retained only for scoring. They are never passed to QReasonCore as
quality signals, preventing oracle leakage in the blind benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import QReasonCore
from .synthetic import top1_accuracy


@dataclass(frozen=True)
class BlindTaskBatch:
    amplitudes: Tensor
    labels: Tensor
    task_name: str


def _batch(
    *,
    batch_size: int,
    state_dim: int,
    seed: int,
    task_name: str,
) -> BlindTaskBatch:
    generator = torch.Generator().manual_seed(seed)
    amplitudes = torch.randn(batch_size, state_dim, generator=generator)
    labels = torch.randint(state_dim, (batch_size,), generator=generator)
    return BlindTaskBatch(amplitudes, labels, task_name)


def make_blind_tasks(
    *,
    batch_size: int = 256,
    state_dim: int = 16,
    seed: int = 29,
) -> list[BlindTaskBatch]:
    """Return labels-only scoring tasks with no verifier signal."""
    return [
        _batch(
            batch_size=batch_size,
            state_dim=state_dim,
            seed=seed,
            task_name="arithmetic_candidates",
        ),
        _batch(
            batch_size=batch_size,
            state_dim=state_dim,
            seed=seed + 1,
            task_name="constraint_candidates",
        ),
        _batch(
            batch_size=batch_size,
            state_dim=state_dim,
            seed=seed + 2,
            task_name="graph_candidates",
        ),
        _batch(
            batch_size=batch_size,
            state_dim=state_dim,
            seed=seed + 3,
            task_name="syntax_candidates",
        ),
    ]


@torch.no_grad()
def evaluate_blind_tasks(
    core: QReasonCore,
    tasks: list[BlindTaskBatch],
) -> list[dict[str, float | str]]:
    """Evaluate QReason with quality omitted entirely."""
    results: list[dict[str, float | str]] = []
    for task in tasks:
        quality = torch.zeros_like(task.amplitudes)
        output = core(task.amplitudes, quality)
        results.append({
            "task": task.task_name,
            "qreason_top1": top1_accuracy(output.scores, task.labels),
            "chance_top1": 1.0 / task.amplitudes.shape[-1],
            "finite": float(torch.isfinite(output.scores).all()),
            "unitary_error": float(output.unitary_error),
            "norm_error": float(output.norm_error),
        })
    return results
