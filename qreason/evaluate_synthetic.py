"""Run the mechanism-level synthetic QReason evaluation."""

from __future__ import annotations

import json

import torch

from .core import QReasonConfig, QReasonCore
from .synthetic import evaluate_core, evaluate_uniform_baseline, make_constraint_batch


def main() -> None:
    batch = make_constraint_batch(batch_size=1024, state_dim=16, seed=17)
    torch.manual_seed(17)
    core = QReasonCore(QReasonConfig(state_dim=16, cycles=4, quality_strength=2.0))
    report = {
        "baseline": evaluate_uniform_baseline(batch),
        "qreason": evaluate_core(core, batch),
        "qreason_disabled": evaluate_core(core, batch, enabled=False),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
