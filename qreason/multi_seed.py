"""Multi-seed, held-out comparison for QReason and a size-matched control."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

import torch

from .core import QReasonConfig
from .trainable import (
    QReasonClassifier,
    RealClassifier,
    evaluate_classifier,
    make_feature_split,
    parameter_count,
    train_classifier,
)


@dataclass(frozen=True)
class MultiSeedReport:
    qreason_accuracy: list[float]
    real_accuracy: list[float]
    qreason_loss: list[float]
    real_loss: list[float]
    qreason_parameters: int
    real_parameters: int

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "qreason": {
                "accuracy": self._summary(self.qreason_accuracy),
                "cross_entropy": self._summary(self.qreason_loss),
                "parameters": self.qreason_parameters,
            },
            "real_baseline": {
                "accuracy": self._summary(self.real_accuracy),
                "cross_entropy": self._summary(self.real_loss),
                "parameters": self.real_parameters,
            },
            "accuracy_delta_mean": mean(self.qreason_accuracy) - mean(self.real_accuracy),
            "parameter_delta": self.qreason_parameters - self.real_parameters,
            "seeds": len(self.qreason_accuracy),
        }


def run_multi_seed(
    seeds: tuple[int, ...] = (3, 5, 7, 11, 13),
) -> MultiSeedReport:
    qreason_accuracy: list[float] = []
    real_accuracy: list[float] = []
    qreason_loss: list[float] = []
    real_loss: list[float] = []
    qreason_parameters = 0
    real_parameters = 0

    for seed in seeds:
        torch.manual_seed(seed)
        split = make_feature_split(
            train_samples=1024,
            test_samples=512,
            feature_dim=24,
            state_dim=8,
            seed=41,
        )
        config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
        qreason = QReasonClassifier(24, config)
        real = RealClassifier(24, config.state_dim, hidden_dim=20)
        train_classifier(qreason, split.train, steps=200, learning_rate=2e-3)
        train_classifier(real, split.train, steps=200, learning_rate=2e-3)
        q_metrics = evaluate_classifier(qreason, split.test)
        r_metrics = evaluate_classifier(real, split.test)
        qreason_accuracy.append(q_metrics["top1_accuracy"])
        real_accuracy.append(r_metrics["top1_accuracy"])
        qreason_loss.append(q_metrics["cross_entropy"])
        real_loss.append(r_metrics["cross_entropy"])
        qreason_parameters = parameter_count(qreason)
        real_parameters = parameter_count(real)

    return MultiSeedReport(
        qreason_accuracy,
        real_accuracy,
        qreason_loss,
        real_loss,
        qreason_parameters,
        real_parameters,
    )
