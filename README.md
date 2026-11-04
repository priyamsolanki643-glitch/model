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
- a held-out bilinear latent-hypothesis comparison with classical controls;
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

`StateVectorEngine` adds an explicit finite computational register with
`2**qubits` complex amplitudes. It applies a model-derived phase oracle and
Hadamard interference cycles before measurement. This is the exact finite
state-vector algebra on classical tensors; its memory and compute grow
exponentially with the qubit count, and it has no physical quantum speedup.
`max_amplitudes` prevents accidental unbounded allocations.

The V1 benchmark is runnable with:

```bash
python -m qreason.evaluate_wave_task
```

It compares wave, real feed-forward, and recurrent classifiers over the same
held-out rule. Results are exploratory and do not establish an advantage.

## Model integration

`attach_qreason(module, config)` registers a shape-preserving forward hook on a
transformer-style block. The block may return a hidden tensor or a tuple whose
first element is the hidden tensor; remaining cache outputs are preserved.

```python
from qreason import WaveStateConfig, attach_qreason

handle = attach_qreason(
    model.model.layers[0],
    WaveStateConfig(hidden_dim=model.config.hidden_size, hypotheses=8),
)
```

The base model is not modified or downloaded by this package. The returned
adapter is independently trainable; remove the hook with `handle.remove()`.
Initial residual output is zero, so attaching it does not perturb a frozen
model before adapter training.

## Kaggle Qwen runner

The repository includes an adapter-only runner for Qwen2.5-0.5B and
Qwen2.5-Coder-1.5B. It requires a Kaggle GPU and downloads weights at runtime:

```bash
pip install transformers accelerate
python -m qreason.qwen_kaggle \
  --model qwen-0.5b \
  --prompt-file prompts.jsonl \
  --layers -1 \
  --qubits 5 \
  --train-steps 0
```

Each JSONL record must contain `prompt` and may contain `completion`. The base
model is frozen; only adapter parameters train when `--train-steps` is
positive. The runner reports baseline and adapted generations, but generation
differences are not evidence of better reasoning. A real evaluation needs
held-out planning tasks with correctness and calibration scoring.

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
