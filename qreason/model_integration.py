"""Attach QReason to generic transformer-style hidden-state modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .wave_state import QReasonAdapter, WaveStateConfig, WaveStateOutput


@dataclass
class QReasonHook:
    """Handle for a registered hidden-state adapter hook."""

    adapter: QReasonAdapter
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
    adapter.eval()

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output if isinstance(output, Tensor) else output[0]
        transformed, _wave = adapter(hidden, enabled=enabled)
        return _replace_hidden(output, transformed)

    handle = module.register_forward_hook(hook)
    return QReasonHook(adapter=adapter, handle=handle)


__all__ = ["QReasonHook", "attach_qreason"]
