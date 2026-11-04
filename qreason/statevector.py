"""Exact finite state-vector quantum-inspired dynamics on classical tensors.

The register stores one complex amplitude for every computational basis state.
This follows the finite state-vector mathematics exactly, but the simulation
cost is exponential in ``qubits`` and provides no physical quantum speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StateVectorConfig:
    hidden_dim: int
    qubits: int = 5
    cycles: int = 2
    max_amplitudes: int = 4096
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.qubits < 1:
            raise ValueError("qubits must be positive")
        if self.cycles < 0:
            raise ValueError("cycles must be non-negative")
        amplitudes = 2**self.qubits
        if amplitudes > self.max_amplitudes:
            raise ValueError(
                f"state vector has {amplitudes} amplitudes; "
                f"max_amplitudes is {self.max_amplitudes}"
            )

    @property
    def amplitudes(self) -> int:
        return 2**self.qubits


@dataclass
class StateVectorOutput:
    amplitudes: Tensor
    probabilities: Tensor
    phase_oracle: Tensor
    measured_scores: Tensor
    norm_error: Tensor
    unitary_error: Tensor


def _normalize(state: Tensor, eps: float) -> Tensor:
    norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    return state / norm.clamp_min(eps)


def _hadamard_transform(state: Tensor) -> Tensor:
    """Apply H to every qubit using pairwise butterfly operations."""
    result = state
    size = state.shape[-1]
    stride = 1
    scale = 2**-0.5
    while stride < size:
        view = result.reshape(*result.shape[:-1], -1, 2, stride)
        first = view[..., 0, :].clone()
        second = view[..., 1, :].clone()
        view[..., 0, :] = (first + second) * scale
        view[..., 1, :] = (first - second) * scale
        stride *= 2
    return result


class StateVectorEngine(nn.Module):
    """Model-conditioned finite register with phase oracle and interference."""

    def __init__(self, config: StateVectorConfig) -> None:
        super().__init__()
        self.config = config
        self.initializer = nn.Linear(config.hidden_dim, 2 * config.amplitudes)
        self.oracle = nn.Linear(config.hidden_dim, config.amplitudes)

    def forward(self, hidden: Tensor, *, enabled: bool = True) -> StateVectorOutput:
        if hidden.ndim < 2 or hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError("hidden must have shape [..., hidden_dim]")
        original_shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, self.config.hidden_dim)
        dimension = self.config.amplitudes
        if not enabled:
            state = torch.zeros(
                flat.shape[0], dimension, dtype=torch.complex64, device=flat.device
            )
            state[:, 0] = 1.0
            probabilities = state.abs().square()
            zeros = torch.zeros((), device=flat.device)
            return StateVectorOutput(
                amplitudes=state.reshape(*original_shape, dimension),
                probabilities=probabilities.reshape(*original_shape, dimension),
                phase_oracle=torch.zeros_like(probabilities),
                measured_scores=probabilities,
                norm_error=zeros,
                unitary_error=zeros,
            )

        raw = self.initializer(flat).reshape(flat.shape[0], dimension, 2)
        state = _normalize(torch.complex(raw[..., 0], raw[..., 1]), self.config.eps)
        oracle_score = torch.tanh(self.oracle(flat))
        phase = torch.pi * oracle_score
        for _ in range(self.config.cycles):
            state = state * torch.exp(1j * phase)
            state = _hadamard_transform(state)
            state = _normalize(state, self.config.eps)

        probabilities = state.abs().square()
        measured_scores = probabilities.clamp_min(self.config.eps).log()
        norm_error = (torch.linalg.vector_norm(state, dim=-1) - 1.0).abs().max()
        return StateVectorOutput(
            amplitudes=state.reshape(*original_shape, dimension),
            probabilities=probabilities.reshape(*original_shape, dimension),
            phase_oracle=phase.reshape(*original_shape, dimension),
            measured_scores=measured_scores.reshape(*original_shape, dimension),
            norm_error=norm_error,
            unitary_error=torch.zeros((), device=flat.device),
        )


__all__ = ["StateVectorConfig", "StateVectorEngine", "StateVectorOutput"]
