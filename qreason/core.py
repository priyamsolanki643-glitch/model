"""A bounded, classical complex-state reasoning core.

The module is intentionally independent of any language model. It provides a
finite hypothesis state, verifier-informed phase marking, Cayley evolution and
normalized measurement scores. It does not simulate physical quantum hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QReasonConfig:
    state_dim: int = 16
    cycles: int = 2
    step_size: float = 0.05
    quality_strength: float = 1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.state_dim < 2:
            raise ValueError("state_dim must be at least 2")
        if self.cycles < 0:
            raise ValueError("cycles must be non-negative")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if self.quality_strength < 0.0:
            raise ValueError("quality_strength must be non-negative")


@dataclass
class QReasonOutput:
    amplitudes: Tensor
    probabilities: Tensor
    scores: Tensor
    phase_angles: Tensor
    unitary_error: Tensor
    norm_error: Tensor


class QReasonCore(nn.Module):
    """Finite complex hypothesis dynamics with explicit safety controls.

    ``amplitudes`` has shape ``[batch, state_dim]`` and may be real or complex.
    ``quality`` has the same shape and is expected to lie in ``[-1, 1]``.
    Positive quality increases the final verifier-coupled score; it is not
    treated as proof of correctness.
    """

    def __init__(self, config: QReasonConfig = QReasonConfig()) -> None:
        super().__init__()
        self.config = config
        dim = config.state_dim
        self.generator_factor = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.phase_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("identity", torch.eye(dim, dtype=torch.complex64))

    def _normalize(self, state: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True)
        return state / norm.clamp_min(self.config.eps)

    def _hermitian(self) -> Tensor:
        factor = torch.complex(self.generator_factor, torch.zeros_like(self.generator_factor))
        return factor @ factor.conj().transpose(-2, -1)

    def unitary(self) -> Tensor:
        hamiltonian = self._hermitian()
        identity = self.identity.to(device=hamiltonian.device, dtype=hamiltonian.dtype)
        half_step = 0.5 * self.config.step_size
        left = identity + 1j * half_step * hamiltonian
        right = identity - 1j * half_step * hamiltonian
        return torch.linalg.solve(left, right)

    def phase_mark(self, state: Tensor, quality: Tensor) -> tuple[Tensor, Tensor]:
        if quality.shape != state.shape:
            raise ValueError("quality must have the same shape as amplitudes")
        quality = quality.to(device=state.device, dtype=state.real.dtype).clamp(-1.0, 1.0)
        angles = torch.pi * torch.tanh(self.phase_scale) * quality
        return state * torch.exp(1j * angles), angles

    def forward(
        self,
        amplitudes: Tensor,
        quality: Tensor | None = None,
        *,
        enabled: bool = True,
    ) -> QReasonOutput:
        if amplitudes.ndim != 2 or amplitudes.shape[-1] != self.config.state_dim:
            raise ValueError("amplitudes must have shape [batch, state_dim]")
        state = amplitudes.to(dtype=torch.complex64)
        if quality is None:
            quality = torch.zeros_like(state.real)
        if not enabled:
            state = self._normalize(state)
            probabilities = state.abs().square()
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
            zeros = torch.zeros_like(probabilities)
            return QReasonOutput(state, probabilities, probabilities.log(), zeros, zeros.max(), zeros.max())

        state = self._normalize(state)
        state, angles = self.phase_mark(state, quality)
        operator = self.unitary()
        for _ in range(self.config.cycles):
            state = torch.matmul(state, operator.transpose(-2, -1))
            state = self._normalize(state)

        probabilities = state.abs().square()
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        quality_bias = self.config.quality_strength * quality.to(probabilities.dtype)
        scores = probabilities.clamp_min(self.config.eps).log() + quality_bias
        unitary_error = torch.linalg.matrix_norm(
            operator.conj().transpose(-2, -1) @ operator - self.identity.to(operator)
        )
        norm_error = (torch.linalg.vector_norm(state, dim=-1) - 1.0).abs().max()
        return QReasonOutput(state, probabilities, scores, angles, unitary_error, norm_error)
