"""Trainable hypothesis selection around the standalone QReason core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .core import QReasonConfig, QReasonCore
from .synthetic import top1_accuracy


@dataclass(frozen=True)
class FeatureTask:
    features: Tensor
    labels: Tensor


@dataclass(frozen=True)
class FeatureTaskSplit:
    train: FeatureTask
    test: FeatureTask


def _make_features(
    *,
    samples: int,
    feature_dim: int,
    state_dim: int,
    generator: torch.Generator,
    target_projection: Tensor,
) -> FeatureTask:
    features = torch.randn(samples, feature_dim, generator=generator)
    labels = (features @ target_projection).argmax(dim=-1)
    return FeatureTask(features, labels)


def make_feature_task(
    *,
    samples: int = 512,
    feature_dim: int = 24,
    state_dim: int = 8,
    seed: int = 41,
) -> FeatureTask:
    """Create a learnable task with a fixed hidden feature-to-candidate rule."""
    if samples < 1 or feature_dim < 1 or state_dim < 2:
        raise ValueError("samples and feature_dim must be positive; state_dim must be >= 2")
    generator = torch.Generator().manual_seed(seed)
    target_projection = torch.randn(feature_dim, state_dim, generator=generator)
    return _make_features(
        samples=samples,
        feature_dim=feature_dim,
        state_dim=state_dim,
        generator=generator,
        target_projection=target_projection,
    )


def make_feature_split(
    *,
    train_samples: int = 512,
    test_samples: int = 256,
    feature_dim: int = 24,
    state_dim: int = 8,
    seed: int = 41,
) -> FeatureTaskSplit:
    """Create train/test data governed by the same hidden task rule."""
    if train_samples < 1 or test_samples < 1:
        raise ValueError("train_samples and test_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    target_projection = torch.randn(feature_dim, state_dim, generator=generator)
    return FeatureTaskSplit(
        train=_make_features(
            samples=train_samples,
            feature_dim=feature_dim,
            state_dim=state_dim,
            generator=generator,
            target_projection=target_projection,
        ),
        test=_make_features(
            samples=test_samples,
            feature_dim=feature_dim,
            state_dim=state_dim,
            generator=generator,
            target_projection=target_projection,
        ),
    )


class QReasonClassifier(nn.Module):
    """Maps features to QReason amplitudes and learned quality estimates."""

    def __init__(self, feature_dim: int, config: QReasonConfig) -> None:
        super().__init__()
        self.amplitude = nn.Linear(feature_dim, config.state_dim * 2)
        self.quality = nn.Linear(feature_dim, config.state_dim)
        self.core = QReasonCore(config)

    def forward(self, features: Tensor) -> Tensor:
        raw = self.amplitude(features)
        amplitudes = torch.complex(raw[..., : self.core.config.state_dim],
                                   raw[..., self.core.config.state_dim:])
        quality = torch.tanh(self.quality(features))
        return self.core(amplitudes, quality).scores


class RealClassifier(nn.Module):
    """Parameter-matched classical control without complex evolution."""

    def __init__(self, feature_dim: int, state_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or state_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


def train_classifier(
    model: nn.Module,
    task: FeatureTask,
    *,
    steps: int = 200,
    learning_rate: float = 2e-3,
) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(task.features), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def evaluate_classifier(model: nn.Module, task: FeatureTask) -> dict[str, float]:
    scores = model(task.features)
    return {
        "top1_accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(nn.functional.cross_entropy(scores, task.labels)),
        "finite": float(torch.isfinite(scores).all()),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
