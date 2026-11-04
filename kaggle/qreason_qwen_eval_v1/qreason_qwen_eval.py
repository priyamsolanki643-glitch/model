"""Kaggle script kernel for QReason adapter evaluation on two Qwen hosts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path("/kaggle/working")
REPOSITORY = WORKSPACE / "qreason-model"
PROMPTS = Path("/kaggle/input/qreason-qwen-eval-v1/prompts.jsonl")
EMBEDDED_PROMPTS = """\
{"prompt":"You are solving a planning problem. Design a reliable step-by-step plan for implementing a rate-limited REST API client with retries, timeouts, idempotency, and tests.","completion":""}
{"prompt":"You are solving a software engineering problem. Explain how to debug a production memory leak in a Python service, including measurements, hypotheses, experiments, and rollback criteria.","completion":""}
{"prompt":"You are solving an algorithm problem. Plan an efficient solution for finding the shortest path in a weighted graph with non-negative edge weights, and state the invariant that proves correctness.","completion":""}
{"prompt":"You are reviewing a code change. Create a structured checklist for detecting data races, resource leaks, missing error handling, and backwards-compatibility risks before merge.","completion":""}
{"prompt":"You are designing a safe migration. Plan how to migrate a large relational database column with zero downtime, verification queries, monitoring, and rollback.","completion":""}
"""


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install_and_clone() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"])
    if REPOSITORY.exists():
        run(["git", "pull", "--ff-only"], cwd=REPOSITORY)
    else:
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/priyamsolanki643-glitch/model.git",
            str(REPOSITORY),
        ])
    sys.path.insert(0, str(REPOSITORY))


def prompt_file() -> Path:
    if PROMPTS.exists():
        return PROMPTS
    target = WORKSPACE / "qreason-prompts.jsonl"
    target.write_text(EMBEDDED_PROMPTS, encoding="utf-8")
    return target


def run_model(model_name: str, prompts: Path) -> dict[str, object]:
    output = WORKSPACE / f"{model_name.replace('/', '_')}-result.json"
    command = [
        sys.executable,
        "-m",
        "qreason.qwen_kaggle",
        "--model",
        model_name,
        "--prompt-file",
        str(prompts),
        "--layers",
        "-1",
        "--qubits",
        "5",
        "--train-steps",
        "50",
        "--checkpoint",
        f"/kaggle/working/{model_name.replace('/', '_')}-qreason-adapter.pt",
        "--device",
        "cuda",
    ]
    with output.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPOSITORY,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return {
        "model": model_name,
        "result_file": str(output),
        "status": "completed",
    }


def main() -> None:
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("Warning: KAGGLE_KERNEL_RUN_TYPE is not set; continuing for local diagnostics.")
    install_and_clone()
    prompts = prompt_file()
    results = [
        run_model("qwen-0.5b", prompts),
        run_model("qwen-coder-1.5b", prompts),
    ]
    summary = WORKSPACE / "qreason-qwen-eval-summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
