"""Five-seed parameter-matched raw Boolean comparison."""

from __future__ import annotations

from statistics import mean, stdev

import torch

from .core import QReasonConfig
from .raw_boolean import (
    RawQReasonClassifier,
    RawRealClassifier,
    evaluate_raw,
    make_raw_boolean_task,
    train_raw_model,
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def run_raw_multi_seed(seeds: tuple[int, ...] = (3, 5, 7, 11, 13)) -> dict[str, object]:
    q_acc: list[float] = []
    r_acc: list[float] = []
    q_loss: list[float] = []
    r_loss: list[float] = []
    q_params = r_params = 0
    for seed in seeds:
        torch.manual_seed(seed)
        train = make_raw_boolean_task(samples=1024, seed=303)
        test = make_raw_boolean_task(samples=512, seed=404)
        config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
        feature_dim = train.candidate_features.shape[-1]
        qreason = RawQReasonClassifier(feature_dim, config)
        real = RawRealClassifier(feature_dim, config.state_dim, hidden_dim=5)
        train_raw_model(qreason, train)
        train_raw_model(real, train)
        q_metrics = evaluate_raw(qreason, test)
        r_metrics = evaluate_raw(real, test)
        q_acc.append(float(q_metrics["top1_accuracy"]))
        r_acc.append(float(r_metrics["top1_accuracy"]))
        q_loss.append(float(q_metrics["cross_entropy"]))
        r_loss.append(float(r_metrics["cross_entropy"]))
        q_params = int(q_metrics["parameters"])
        r_params = int(r_metrics["parameters"])
    return {
        "qreason": {"accuracy": _summary(q_acc), "cross_entropy": _summary(q_loss), "parameters": q_params},
        "real_baseline": {"accuracy": _summary(r_acc), "cross_entropy": _summary(r_loss), "parameters": r_params},
        "accuracy_delta_mean": mean(q_acc) - mean(r_acc),
        "parameter_delta": q_params - r_params,
        "seeds": len(seeds),
    }
