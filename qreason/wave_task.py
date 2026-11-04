"""Controlled latent-hypothesis benchmark for the V1 wave-state engine."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

import torch
from torch import Tensor, nn

from .synthetic import top1_accuracy
from .trainable import parameter_count
from .wave_state import WaveStateConfig, WaveStateEngine


@dataclass(frozen=True)
class WaveTask:
    features: Tensor
    labels: Tensor


@dataclass(frozen=True)
class WaveTaskSplit:
    train: WaveTask
    test: WaveTask


def _labels(features: Tensor, projection: Tensor) -> Tensor:
    first = features[..., : projection.shape[0]]
    second = features[..., projection.shape[0] :]
    scores = torch.einsum("bi,ijk,bj->bk", first, projection, second)
    return scores.argmax(dim=-1)


def make_wave_task_split(
    *,
    train_samples: int = 1024,
    test_samples: int = 512,
    hidden_dim: int = 12,
    hypotheses: int = 8,
    seed: int = 1201,
) -> WaveTaskSplit:
    """Create a held-out bilinear rule without exposing labels to the models."""
    if hidden_dim < 4 or hidden_dim % 2:
        raise ValueError("hidden_dim must be an even number of at least 4")
    if min(train_samples, test_samples, hypotheses) < 1:
        raise ValueError("sample counts and hypotheses must be positive")
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        hidden_dim // 2, hidden_dim // 2, hypotheses, generator=generator
    )
    features = torch.randn(
        train_samples + test_samples, hidden_dim, generator=generator
    )
    labels = _labels(features, projection)
    return WaveTaskSplit(
        train=WaveTask(features[:train_samples], labels[:train_samples]),
        test=WaveTask(features[train_samples:], labels[train_samples:]),
    )


class WaveClassifier(nn.Module):
    def __init__(self, hidden_dim: int, hypotheses: int) -> None:
        super().__init__()
        self.engine = WaveStateEngine(
            WaveStateConfig(hidden_dim=hidden_dim, hypotheses=hypotheses, cycles=2)
        )

    def forward(self, features: Tensor) -> Tensor:
        output = self.engine(features)
        return output.probabilities.clamp_min(1e-8).log()


class RealClassifier(nn.Module):
    def __init__(self, hidden_dim: int, hypotheses: int, width: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.Tanh(),
            nn.Linear(width, hypotheses),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class RecurrentClassifier(nn.Module):
    def __init__(self, hidden_dim: int, hypotheses: int, width: int = 16) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_dim, width)
        self.recurrent = nn.GRUCell(width, width)
        self.output = nn.Linear(width, hypotheses)

    def forward(self, features: Tensor) -> Tensor:
        encoded = torch.tanh(self.input(features))
        hidden = torch.zeros_like(encoded)
        hidden = self.recurrent(encoded, hidden)
        return self.output(hidden)


def train_wave_model(
    model: nn.Module,
    task: WaveTask,
    *,
    steps: int = 300,
    learning_rate: float = 2e-3,
) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(task.features), task.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def evaluate_wave_model(model: nn.Module, task: WaveTask) -> dict[str, float]:
    scores = model(task.features)
    return {
        "accuracy": top1_accuracy(scores, task.labels),
        "cross_entropy": float(nn.functional.cross_entropy(scores, task.labels)),
        "finite": float(torch.isfinite(scores).all()),
    }


def _closest_width(
    factory: type[nn.Module],
    hidden_dim: int,
    hypotheses: int,
    target_parameters: int,
) -> int:
    candidates = range(1, 128)
    return min(
        candidates,
        key=lambda width: abs(
            parameter_count(factory(hidden_dim, hypotheses, width))
            - target_parameters
        ),
    )


def run_wave_comparison(
    seeds: tuple[int, ...] = (3, 5, 7, 11, 13),
) -> dict[str, object]:
    accuracies: dict[str, list[float]] = {"wave": [], "real": [], "recurrent": []}
    losses: dict[str, list[float]] = {"wave": [], "real": [], "recurrent": []}
    counts: dict[str, int] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        split = make_wave_task_split()
        wave = WaveClassifier(12, 8)
        real_width = _closest_width(RealClassifier, 12, 8, parameter_count(wave))
        recurrent_width = _closest_width(
            RecurrentClassifier, 12, 8, parameter_count(wave)
        )
        models = {
            "wave": wave,
            "real": RealClassifier(12, 8, real_width),
            "recurrent": RecurrentClassifier(12, 8, recurrent_width),
        }
        for name, model in models.items():
            train_wave_model(model, split.train)
            metrics = evaluate_wave_model(model, split.test)
            accuracies[name].append(metrics["accuracy"])
            losses[name].append(metrics["cross_entropy"])
            counts[name] = parameter_count(model)
    return {
        name: {
            "accuracy": {"mean": mean(accuracies[name]), "std": stdev(accuracies[name])},
            "cross_entropy": {"mean": mean(losses[name]), "std": stdev(losses[name])},
            "parameters": counts[name],
        }
        for name in accuracies
    } | {"seeds": len(seeds), "chance": 1.0 / 8.0}


__all__ = [
    "RealClassifier",
    "RecurrentClassifier",
    "WaveClassifier",
    "WaveTask",
    "WaveTaskSplit",
    "evaluate_wave_model",
    "make_wave_task_split",
    "run_wave_comparison",
    "train_wave_model",
]
