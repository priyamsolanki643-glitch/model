"""Train and evaluate QReason on Boolean constraint candidate selection."""

from __future__ import annotations

import json

import torch

from .boolean_tasks import (
    CandidateQReasonClassifier,
    CandidateRealClassifier,
    evaluate_boolean,
    make_boolean_task,
    train_boolean_model,
)
from .core import QReasonConfig
from .trainable import parameter_count


def main() -> None:
    torch.manual_seed(101)
    train = make_boolean_task(samples=1024, seed=101)
    test = make_boolean_task(samples=512, seed=202)
    config = QReasonConfig(state_dim=8, cycles=2, quality_strength=1.0)
    qreason = CandidateQReasonClassifier(train.candidates.shape[-1], config)
    real = CandidateRealClassifier(train.candidates.shape[-1], config.state_dim)
    train_boolean_model(qreason, train)
    train_boolean_model(real, train)
    print(json.dumps({
        "qreason": evaluate_boolean(qreason, test),
        "real_baseline": evaluate_boolean(real, test),
        "parameters": {
            "qreason": parameter_count(qreason),
            "real_baseline": parameter_count(real),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
