"""Run structured Boolean controls with shared candidate-clause features."""

from __future__ import annotations

import json

from .structured_boolean import run_structured_comparison


def main() -> None:
    print(json.dumps(run_structured_comparison(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
