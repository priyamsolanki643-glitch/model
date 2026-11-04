"""Learned finite wave-state dynamics for model-agnostic hidden states.

This is a classical tensor implementation of a finite complex hypothesis basis.
It is not a physical quantum simulator and does not provide quantum speedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class WaveStateConfig:
    hidden_dim: int
    hypotheses: int = 8
    cycles: int = 2
    step_size: float = 0.05
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.hypotheses < 2:
            raise ValueError("hypotheses must be at least 2")
        if self.cycles < 0:
            raise ValueError("cycles must be non-negative")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")


@dataclass
class WaveStateOutput:
    state: Tensor
    probabilities: Tensor
    quality: Tensor
    collapsed: Tensor
    unitary_error: Tensor
    norm_error: Tensor


class WaveStateEngine(nn.Module):
    """Generate, evolve, and measure a finite complex hypothesis wave."""

    def __init__(self, config: WaveStateConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hypotheses
        self.hypothesis = nn.Linear(config.hidden_dim, 2 * dim * config.hidden_dim)
        self.quality = nn.Linear(config.hidden_dim, dim)
        self.generator_factor = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.phase_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("identity", torch.eye(dim, dtype=torch.complex64))

    def _normalize(self, state: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(state, dim=(-2, -1), keepdim=True)
        return state / norm.clamp_min(self.config.eps)

    def _operator(self, quality: Tensor) -> Tensor:
        factor = torch.complex(self.generator_factor, torch.zeros_like(self.generator_factor))
        base = factor @ factor.conj().transpose(-2, -1)
        identity = self.identity.to(device=base.device, dtype=base.dtype)
        diagonal = torch.diag_embed(quality.to(base.real.dtype))
        hamiltonian = base + diagonal.to(base.dtype)
        batch_identity = identity.expand(quality.shape[0], -1, -1)
        half_step = 0.5 * self.config.step_size
        left = batch_identity + 1j * half_step * hamiltonian
        right = batch_identity - 1j * half_step * hamiltonian
        return torch.linalg.solve(left, right)

    def forward(self, hidden: Tensor, *, enabled: bool = True) -> WaveStateOutput:
        if hidden.ndim < 2 or hidden.shape[-1] != self.config.hidden_dim:
            raise ValueError("hidden must have shape [..., hidden_dim]")
        original_shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, self.config.hidden_dim)
        if not enabled:
            zeros = torch.zeros(
                flat.shape[0], self.config.hypotheses, device=flat.device, dtype=flat.dtype
            )
            return WaveStateOutput(
                state=torch.zeros(
                    flat.shape[0], self.config.hypotheses, self.config.hidden_dim,
                    device=flat.device, dtype=torch.complex64,
                ),
                probabilities=torch.full(
                    (flat.shape[0], self.config.hypotheses),
                    1.0 / self.config.hypotheses,
                    device=flat.device, dtype=flat.dtype,
                ),
                quality=zeros,
                collapsed=flat,
                unitary_error=torch.zeros((), device=flat.device),
                norm_error=torch.zeros((), device=flat.device),
            )

        raw = self.hypothesis(flat).reshape(
            flat.shape[0], self.config.hypotheses, self.config.hidden_dim, 2
        )
        state = torch.complex(raw[..., 0], raw[..., 1])
        state = self._normalize(state)
        quality = torch.tanh(self.quality(flat))
        angles = torch.pi * torch.tanh(self.phase_scale) * quality
        state = state * torch.exp(1j * angles.unsqueeze(-1))
        operator = self._operator(quality)
        for _ in range(self.config.cycles):
            state = torch.einsum("bij,bjd->bid", operator, state)
            state = self._normalize(state)

        probabilities = state.abs().square().sum(dim=-1)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        collapsed = torch.einsum(
            "bk,bkd->bd", probabilities.to(state.dtype), state
        ).real
        collapsed = collapsed / torch.linalg.vector_norm(
            collapsed, dim=-1, keepdim=True
        ).clamp_min(self.config.eps)
        identity = self.identity.to(device=operator.device, dtype=operator.dtype)
        unitary_error = torch.linalg.matrix_norm(
            operator.conj().transpose(-2, -1) @ operator - identity
        ).max()
        norm_error = (
            torch.linalg.vector_norm(state, dim=(-2, -1)) - 1.0
        ).abs().max()
        return WaveStateOutput(
            state=state.reshape(*original_shape, self.config.hypotheses, self.config.hidden_dim),
            probabilities=probabilities.reshape(*original_shape, self.config.hypotheses),
            quality=quality.reshape(*original_shape, self.config.hypotheses),
            collapsed=collapsed.reshape(*original_shape, self.config.hidden_dim),
            unitary_error=unitary_error,
            norm_error=norm_error,
        )


class QReasonAdapter(nn.Module):
    """Shape-preserving residual adapter for arbitrary hidden-state tensors."""

    def __init__(self, config: WaveStateConfig) -> None:
        super().__init__()
        self.engine = WaveStateEngine(config)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden: Tensor, *, enabled: bool = True) -> tuple[Tensor, WaveStateOutput]:
        if not enabled:
            output = self.engine(hidden, enabled=False)
            return hidden, output
        wave = self.engine(hidden)
        update = self.output(wave.collapsed)
        update = update.reshape(-1, hidden.shape[-1])
        result = hidden.reshape(-1, hidden.shape[-1]) + torch.tanh(self.residual_gate) * update
        return result.reshape_as(hidden), wave
