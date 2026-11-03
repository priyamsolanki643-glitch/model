"""Evaluate QReason on guaranteed-unique raw Boolean constraints."""

from __future__ import annotations

import json
from statistics import mean, stdev

import torch

from .core import QReasonConfig
from .raw_boolean import RawQReasonClassifier, RawRealClassifier
from .unique_boolean import (
    evaluate_unique,
    make_unique_boolean_task,
    train_unique_model,
    verify_unique,
)


def main() -> None:
    q_acc: list[float] = []
    r_acc: list[float] = []
    for seed in (3, 5, 7, 11, 13):
        torch.manual_seed(seed)
        train = make_unique_boolean_task(samples=1024, seed=501)
        test = make_unique_boolean_task(samples=512, seed=601)
        if not verify_unique(train) or not verify_unique(test):
            raise RuntimeError("unique-solution task generation failed")
        config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
        feature_dim = train.candidate_features.shape[-1]
        qreason = RawQReasonClassifier(feature_dim, config)
        real = RawRealClassifier(feature_dim, 8, hidden_dim=5)
        train_unique_model(qreason, train)
        train_unique_model(real, train)
        q_acc.append(float(evaluate_unique(qreason, test)["top1_accuracy"]))
        r_acc.append(float(evaluate_unique(real, test)["top1_accuracy"]))
    print(json.dumps({
        "qreason_accuracy": {"mean": mean(q_acc), "std": stdev(q_acc)},
        "real_baseline_accuracy": {"mean": mean(r_acc), "std": stdev(r_acc)},
        "accuracy_delta_mean": mean(q_acc) - mean(r_acc),
        "seeds": 5,
        "candidate_count": 8,
        "unique_valid_solution": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
