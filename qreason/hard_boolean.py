"""Harder near-miss Boolean candidate benchmark with parameter-matched controls."""

from __future__ import annotations

from statistics import mean, stdev

import torch

from .core import QReasonConfig
from .structured_boolean import (
    StructuredQReason,
    StructuredRealControl,
    StructuredRecurrentControl,
    encode_candidate_clause_interactions,
    evaluate_structured,
    train_structured,
)
from .unique_boolean import UniqueBooleanTask, verify_unique
from .trainable import parameter_count


def make_hard_boolean_task(
    *,
    samples: int = 512,
    candidate_count: int = 8,
    variable_count: int = 10,
    clause_count: int = 16,
    seed: int = 1001,
) -> UniqueBooleanTask:
    """Create unique tasks where each distractor is a near-miss assignment."""
    if candidate_count < 2 or variable_count < 3 or clause_count < 1:
        raise ValueError("task dimensions are too small")
    generator = torch.Generator().manual_seed(seed)
    candidates = torch.randint(
        0, 2, (samples, candidate_count, variable_count), generator=generator
    ).float()
    labels = torch.randint(candidate_count, (samples,), generator=generator)
    clauses = torch.empty(samples, clause_count, 3, dtype=torch.long)
    for sample in range(samples):
        target_index = int(labels[sample])
        target = candidates[sample, target_index]
        for clause_index in range(clause_count):
            variables = torch.randperm(variable_count, generator=generator)[:3]
            clauses[sample, clause_index] = torch.tensor([
                int(variable + 1) if int(target[variable]) else -int(variable + 1)
                for variable in variables
            ])
        distractor_rank = 0
        for candidate_index in range(candidate_count):
            if candidate_index == target_index:
                continue
            # Make distractors close to the target by flipping all literals
            # in one random clause, which guarantees that clause is violated.
            candidates[sample, candidate_index] = target
            clause_index = distractor_rank % clause_count
            flip_variables = clauses[sample, clause_index].abs() - 1
            candidates[sample, candidate_index, flip_variables] = 1.0 - target[flip_variables]
            distractor_rank += 1
    task = UniqueBooleanTask(candidates, clauses, labels)
    if not verify_unique(task):
        raise RuntimeError("hard task generator failed the unique-solution invariant")
    return task


def closest_real_width(feature_dim: int, target_parameters: int) -> int:
    """Find a small real-control width closest to a target parameter count."""
    candidates = range(1, 128)
    return min(
        candidates,
        key=lambda width: abs(
            parameter_count(StructuredRealControl(feature_dim, 8, width))
            - target_parameters
        ),
    )


def run_hard_comparison(seeds: tuple[int, ...] = (3, 5, 7, 11, 13)) -> dict[str, object]:
    accuracies = {"qreason": [], "real": [], "recurrent": []}
    losses = {"qreason": [], "real": [], "recurrent": []}
    counts: dict[str, int] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        train = encode_candidate_clause_interactions(
            make_hard_boolean_task(samples=1024, seed=1001)
        )
        test = encode_candidate_clause_interactions(
            make_hard_boolean_task(samples=512, seed=1101)
        )
        feature_dim = train.features.shape[-1]
        qreason = StructuredQReason(
            feature_dim, QReasonConfig(state_dim=8, cycles=2), width=8
        )
        real_width = closest_real_width(feature_dim, parameter_count(qreason))
        real = StructuredRealControl(feature_dim, 8, width=real_width)
        recurrent = StructuredRecurrentControl(feature_dim, 8, width=4)
        models = {"qreason": qreason, "real": real, "recurrent": recurrent}
        for name, model in models.items():
            train_structured(model, train, steps=350)
            metrics = evaluate_structured(model, test)
            accuracies[name].append(metrics["accuracy"])
            losses[name].append(metrics["cross_entropy"])
            counts[name] = parameter_count(model)
    return {
        name: {
            "accuracy": {"mean": mean(accuracies[name]), "std": stdev(accuracies[name])},
            "cross_entropy": {"mean": mean(losses[name]), "std": stdev(losses[name])},
            "parameters": counts[name],
        }
        for name in models
    } | {"seeds": len(seeds), "feature_dim": feature_dim}
