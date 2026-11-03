"""Train QReason and a real-valued control on the same synthetic task."""

from __future__ import annotations

import json

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


def main() -> None:
    torch.manual_seed(41)
    split = make_feature_split(
        train_samples=1024,
        test_samples=512,
        feature_dim=24,
        state_dim=8,
        seed=41,
    )
    config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
    qreason = QReasonClassifier(24, config)
    real = RealClassifier(24, config.state_dim)
    q_losses = train_classifier(qreason, split.train)
    r_losses = train_classifier(real, split.train)
    print(json.dumps({
        "qreason": evaluate_classifier(qreason, split.test),
        "real_baseline": evaluate_classifier(real, split.test),
        "parameter_count": {
            "qreason": parameter_count(qreason),
            "real_baseline": parameter_count(real),
        },
        "initial_loss": {
            "qreason": q_losses[0],
            "real_baseline": r_losses[0],
        },
        "final_loss": {
            "qreason": q_losses[-1],
            "real_baseline": r_losses[-1],
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
