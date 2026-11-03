"""Run the parameter-matched raw Boolean comparison."""

from __future__ import annotations

import json

from .raw_boolean_multi_seed import run_raw_multi_seed


def main() -> None:
    print(json.dumps(run_raw_multi_seed(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
