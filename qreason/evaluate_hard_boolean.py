"""Run the harder near-miss Boolean comparison."""

from __future__ import annotations

import json

from .hard_boolean import run_hard_comparison


def main() -> None:
    print(json.dumps(run_hard_comparison(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
