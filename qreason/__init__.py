"""Standalone quantum-inspired reasoning primitives."""

from .core import QReasonConfig, QReasonCore, QReasonOutput
from .model_integration import (
    QReasonHook,
    StateVectorAdapter,
    attach_qreason,
    attach_statevector_qreason,
)
from .statevector import StateVectorConfig, StateVectorEngine, StateVectorOutput
from .wave_state import QReasonAdapter, WaveStateConfig, WaveStateEngine, WaveStateOutput

__all__ = [
    "QReasonAdapter",
    "QReasonConfig",
    "QReasonCore",
    "QReasonOutput",
    "QReasonHook",
    "StateVectorConfig",
    "StateVectorEngine",
    "StateVectorOutput",
    "StateVectorAdapter",
    "WaveStateConfig",
    "WaveStateEngine",
    "WaveStateOutput",
    "attach_qreason",
    "attach_statevector_qreason",
]
