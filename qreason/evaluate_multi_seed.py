"""Run the five-seed held-out QReason comparison."""

from __future__ import annotations

import json

from .multi_seed import run_multi_seed


def main() -> None:
    print(json.dumps(run_multi_seed().as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
