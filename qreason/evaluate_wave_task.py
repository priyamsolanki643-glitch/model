"""Run the controlled V1 latent-hypothesis comparison."""

from __future__ import annotations

import json

from .wave_task import run_wave_comparison


def main() -> None:
    print(json.dumps(run_wave_comparison(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
