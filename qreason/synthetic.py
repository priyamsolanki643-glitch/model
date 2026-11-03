"""Synthetic hypothesis tasks for mechanism-level QReason evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import QReasonCore


@dataclass(frozen=True)
class SyntheticBatch:
    amplitudes: Tensor
    quality: Tensor
    valid_index: Tensor


def make_constraint_batch(
    *,
    batch_size: int = 128,
    state_dim: int = 16,
    seed: int = 17,
) -> SyntheticBatch:
    """Create one marked-valid hypothesis among noisy alternatives.

    The task is deliberately synthetic: the quality signal is the known
    constraint oracle, allowing us to test state ranking independently of an
    LLM or an untrusted learned verifier.
    """
    if batch_size < 1 or state_dim < 2:
        raise ValueError("batch_size must be positive and state_dim must be >= 2")
    generator = torch.Generator().manual_seed(seed)
    amplitudes = torch.randn(batch_size, state_dim, generator=generator)
    valid_index = torch.randint(state_dim, (batch_size,), generator=generator)
    quality = torch.full((batch_size, state_dim), -1.0)
    quality.scatter_(1, valid_index.unsqueeze(1), 1.0)
    return SyntheticBatch(amplitudes, quality, valid_index)


def top1_accuracy(scores: Tensor, valid_index: Tensor) -> float:
    if scores.ndim != 2 or valid_index.ndim != 1 or scores.shape[0] != valid_index.shape[0]:
        raise ValueError("scores must be [batch, states] and valid_index [batch]")
    return float((scores.argmax(dim=-1) == valid_index).float().mean())


@torch.no_grad()
def evaluate_core(
    core: QReasonCore,
    batch: SyntheticBatch,
    *,
    enabled: bool = True,
) -> dict[str, float]:
    output = core(batch.amplitudes, batch.quality, enabled=enabled)
    predicted = output.scores.argmax(dim=-1)
    valid_probability = output.probabilities.gather(
        1, batch.valid_index.unsqueeze(1)
    ).mean()
    invalid_mask = torch.ones_like(output.probabilities, dtype=torch.bool)
    invalid_mask.scatter_(1, batch.valid_index.unsqueeze(1), False)
    invalid_probability = output.probabilities.masked_select(invalid_mask).mean()
    return {
        "top1_accuracy": top1_accuracy(output.scores, batch.valid_index),
        "valid_probability": float(valid_probability),
        "invalid_probability": float(invalid_probability),
        "unitary_error": float(output.unitary_error),
        "norm_error": float(output.norm_error),
        "finite": float(torch.isfinite(output.scores).all()),
        "predicted_count": float(predicted.unique().numel()),
    }


def evaluate_uniform_baseline(batch: SyntheticBatch) -> dict[str, float]:
    """Evaluate an uninformative equal-score baseline."""
    scores = torch.zeros_like(batch.quality)
    probabilities = torch.full_like(scores, 1.0 / scores.shape[-1])
    return {
        "top1_accuracy": top1_accuracy(scores, batch.valid_index),
        "valid_probability": float(probabilities.gather(
            1, batch.valid_index.unsqueeze(1)
        ).mean()),
        "invalid_probability": float(probabilities[:, 1:].mean()),
    }
