"""Standalone quantum-inspired reasoning primitives."""

from .core import QReasonConfig, QReasonCore, QReasonOutput
from .model_integration import QReasonHook, attach_qreason
from .wave_state import QReasonAdapter, WaveStateConfig, WaveStateEngine, WaveStateOutput

__all__ = [
    "QReasonAdapter",
    "QReasonConfig",
    "QReasonCore",
    "QReasonOutput",
    "QReasonHook",
    "WaveStateConfig",
    "WaveStateEngine",
    "WaveStateOutput",
    "attach_qreason",
]
