"""Run the verifier-free QReason sanity benchmark."""

from __future__ import annotations

import json

import torch

from .blind_tasks import evaluate_blind_tasks, make_blind_tasks
from .core import QReasonConfig, QReasonCore


def main() -> None:
    torch.manual_seed(29)
    tasks = make_blind_tasks(batch_size=1024, state_dim=16, seed=29)
    core = QReasonCore(QReasonConfig(state_dim=16, cycles=4))
    print(json.dumps(evaluate_blind_tasks(core, tasks), indent=2))


if __name__ == "__main__":
    main()
