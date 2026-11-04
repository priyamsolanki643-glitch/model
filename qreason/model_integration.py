"""Attach QReason to generic transformer-style hidden-state modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .wave_state import QReasonAdapter, WaveStateConfig, WaveStateOutput
from .statevector import StateVectorConfig, StateVectorEngine


@dataclass
class QReasonHook:
    """Handle for a registered hidden-state adapter hook."""

    adapter: nn.Module
    handle: Any

    def remove(self) -> None:
        self.handle.remove()


def _replace_hidden(output: Any, hidden: Tensor) -> Any:
    if isinstance(output, Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError("target module must return a tensor, tuple, or list")


def attach_qreason(
    module: nn.Module,
    config: WaveStateConfig,
    *,
    enabled: bool = True,
) -> QReasonHook:
    """Register a shape-preserving adapter on a transformer block.

    The hook expects the first module output to be the hidden-state tensor,
    matching common transformer block APIs. The base module remains unchanged.
    """
    adapter = QReasonAdapter(config)
    parameter = next(module.parameters(), None)
    if parameter is not None:
        adapter.to(parameter.device)
    adapter.eval()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, Tensor) else output[0]
        if next(adapter.parameters()).device != hidden.device:
            adapter.to(hidden.device)
        transformed, _wave = adapter(hidden, enabled=enabled)
        return _replace_hidden(output, transformed)

    handle = module.register_forward_hook(hook)
    return QReasonHook(adapter=adapter, handle=handle)


class StateVectorAdapter(nn.Module):
    """Shape-preserving residual adapter backed by an explicit state vector."""

    def __init__(self, config: StateVectorConfig) -> None:
        super().__init__()
        self.engine = StateVectorEngine(config)
        self.output = nn.Linear(config.amplitudes, config.hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden: Tensor, *, enabled: bool = True) -> tuple[Tensor, Any]:
        work_hidden = hidden.float()
        wave = self.engine(work_hidden, enabled=enabled)
        if not enabled:
            return hidden, wave
        probabilities = wave.probabilities.reshape(-1, self.engine.config.amplitudes)
        update = self.output(probabilities).reshape_as(work_hidden)
        update = update.to(dtype=hidden.dtype)
        return hidden + torch.tanh(self.residual_gate) * update, wave


def attach_statevector_qreason(
    module: nn.Module,
    config: StateVectorConfig,
    *,
    enabled: bool = True,
) -> QReasonHook:
    """Register the explicit state-vector adapter on a transformer block."""
    adapter = StateVectorAdapter(config)
    parameter = next(module.parameters(), None)
    if parameter is not None:
        adapter.to(parameter.device)
    adapter.eval()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, Tensor) else output[0]
        if next(adapter.parameters()).device != hidden.device:
            adapter.to(hidden.device)
        transformed, _wave = adapter(hidden, enabled=enabled)
        return _replace_hidden(output, transformed)

    handle = module.register_forward_hook(hook)
    return QReasonHook(adapter=adapter, handle=handle)


__all__ = [
    "QReasonHook",
    "StateVectorAdapter",
    "attach_qreason",
    "attach_statevector_qreason",
]
