"""Kaggle runner for frozen Qwen hosts and trainable QReason adapters.

The base model remains frozen. Adapter parameters are the only trainable
parameters when ``--train-steps`` is positive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model_integration import QReasonHook, attach_statevector_qreason
from .statevector import StateVectorConfig


MODEL_IDS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
}


def load_qwen(model_id: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers in Kaggle before running this module"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def attach_layers(
    model: Any,
    *,
    layer_indices: list[int],
    qubits: int,
    enabled: bool,
) -> list[QReasonHook]:
    layers = model.model.layers
    hidden_dim = model.config.hidden_size
    config = StateVectorConfig(hidden_dim=hidden_dim, qubits=qubits)
    return [
        attach_statevector_qreason(
            layers[index], config, enabled=enabled
        )
        for index in layer_indices
    ]


def _encode(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {name: value.to(device) for name, value in encoded.items()}


@torch.no_grad()
def generate(model: Any, tokenizer: Any, prompt: str, device: torch.device) -> str:
    inputs = _encode(tokenizer, prompt, device)
    output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_loss(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, str]],
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.config.use_cache = False
    for record in records:
        text = record["prompt"] + record.get("completion", "")
        inputs = _encode(tokenizer, text, device)
        with torch.no_grad():
            labels = inputs["input_ids"].clone()
            losses.append(float(model(**inputs, labels=labels).loss))
    return sum(losses) / len(losses)


def train_adapters(
    model: Any,
    tokenizer: Any,
    hooks: list[QReasonHook],
    records: list[dict[str, str]],
    device: torch.device,
    steps: int,
) -> list[float]:
    parameters = [
        parameter
        for hook in hooks
        for parameter in hook.adapter.parameters()
        if parameter.requires_grad
    ]
    for hook in hooks:
        if hasattr(hook.adapter, "residual_gate"):
            hook.adapter.residual_gate.data.fill_(0.1)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    losses: list[float] = []
    model.config.use_cache = False
    for step in range(steps):
        record = records[step % len(records)]
        text = record["prompt"] + record.get("completion", "")
        inputs = _encode(tokenizer, text, device)
        labels = inputs["input_ids"].clone()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def save_adapter_checkpoint(
    hooks: list[QReasonHook],
    path: Path,
    *,
    model_id: str,
    layer_indices: list[int],
    qubits: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_id,
            "layers": layer_indices,
            "qubits": qubits,
            "adapters": [hook.adapter.state_dict() for hook in hooks],
        },
        path,
    )


def load_adapter_checkpoint(hooks: list[QReasonHook], path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    adapter_states = checkpoint.get("adapters")
    if not isinstance(adapter_states, list) or len(adapter_states) != len(hooks):
        raise ValueError("checkpoint adapter count does not match selected layers")
    for hook, state in zip(hooks, adapter_states):
        hook.adapter.load_state_dict(state)
    return checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_id = MODEL_IDS[args.model]
    tokenizer, model = load_qwen(model_id, args.device)
    device = next(model.parameters()).device
    layer_indices = [int(value) for value in args.layers.split(",")]
    records = [
        json.loads(line)
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = [generate(model, tokenizer, item["prompt"], device) for item in records]
    base_loss = evaluate_loss(model, tokenizer, records, device)
    hooks = attach_layers(
        model, layer_indices=layer_indices, qubits=args.qubits, enabled=True
    )
    checkpoint_info: dict[str, Any] | None = None
    if args.resume:
        checkpoint_info = load_adapter_checkpoint(hooks, Path(args.resume))
    train_losses = train_adapters(
        model, tokenizer, hooks, records, device, args.train_steps
    ) if args.train_steps else []
    adapted_loss = evaluate_loss(model, tokenizer, records, device)
    if args.checkpoint:
        save_adapter_checkpoint(
            hooks,
            Path(args.checkpoint),
            model_id=model_id,
            layer_indices=layer_indices,
            qubits=args.qubits,
        )
    adapted = [generate(model, tokenizer, item["prompt"], device) for item in records]
    for hook in hooks:
        hook.remove()
    return {
        "model": model_id,
        "layers": layer_indices,
        "qubits": args.qubits,
        "train_steps": args.train_steps,
        "records": len(records),
        "base_loss": base_loss,
        "adapted_loss": adapted_loss,
        "adapter_loss_start": train_losses[0] if train_losses else None,
        "adapter_loss_end": train_losses[-1] if train_losses else None,
        "checkpoint": args.checkpoint,
        "resumed": checkpoint_info is not None,
        "baseline": baseline,
        "adapted": adapted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS, default="qwen-0.5b")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
