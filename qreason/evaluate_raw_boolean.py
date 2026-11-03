"""Evaluate QReason on raw clauses and candidate assignments."""

from __future__ import annotations

import json

import torch

from .core import QReasonConfig
from .raw_boolean import (
    RawQReasonClassifier,
    RawRealClassifier,
    evaluate_raw,
    make_raw_boolean_task,
    train_raw_model,
)


def main() -> None:
    torch.manual_seed(303)
    train = make_raw_boolean_task(samples=1024, seed=303)
    test = make_raw_boolean_task(samples=512, seed=404)
    config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
    feature_dim = train.candidate_features.shape[-1]
    qreason = RawQReasonClassifier(feature_dim, config)
    real = RawRealClassifier(feature_dim, config.state_dim)
    train_raw_model(qreason, train)
    train_raw_model(real, train)
    print(json.dumps({
        "qreason": evaluate_raw(qreason, test),
        "real_baseline": evaluate_raw(real, test),
        "input_contract": {
            "candidate_features": feature_dim,
            "contains_precomputed_satisfaction": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
