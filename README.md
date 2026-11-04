# QReason

QReason is a model-agnostic, classical PyTorch adapter research project for
testing complex latent amplitudes, phase evolution, and interference-inspired
reasoning dynamics.

This repository intentionally does **not** include a base language model.
The adapter is being developed independently so it can later be attached to a
frozen model through a shape-preserving interface.

## Status

The current implementation is a research foundation, not a quantum computer.
It uses finite complex tensors and classical PyTorch. It does not claim
physical superposition, quantum speedup, guaranteed truth, or automatic
hallucination removal.

Current work includes:

- numerical and gradient invariants for the QReason core;
- a finite `K`-hypothesis complex wave-state engine with shape-preserving adapter;
- oracle-informed synthetic controls;
- verifier-free and trainable synthetic tasks;
- Boolean candidate-selection benchmarks;
- parameter-counted classical and recurrent controls.

The benchmark results are exploratory and must not be interpreted as evidence
of improved language-model reasoning.

## Wave-state V1

`WaveStateEngine` maps each hidden vector to `K` learned complex hypotheses,
applies input-conditioned phase and bounded Cayley evolution, then measures
and collapses the state back to the original hidden width. `QReasonAdapter`
wraps this in a residual, shape-preserving interface with an exact disabled
identity path.

This is a finite classical approximation. `K` is explicit and finite; the
implementation does not materialize an exponentially large physical quantum
state.

## Layout

```text
qreason/
  core.py                 Complex state and bounded evolution
  synthetic.py            Oracle-informed synthetic tasks
  blind_tasks.py          Verifier-free tasks
  trainable.py            Trainable model wrappers
  *_boolean.py            Boolean benchmark variants
  evaluate_*.py           Reproducible benchmark entry points
  tests/                  Numerical and benchmark contract tests
```

## Setup

```bash
python -m pip install torch
python -m unittest discover -s qreason/tests -p "test_*.py"
```

All current tests run without a model download or GPU.

## Research boundary

The next major milestone is a mathematically explicit, model-agnostic
hypothesis-space interface. Qwen or another base model will only be integrated
after the representation, controls, ablations, and evaluation protocol are
locked.
