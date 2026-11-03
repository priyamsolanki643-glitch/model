# QReason Core

This package is the first clean-engineering milestone for the project: a
standalone, classical quantum-inspired hypothesis-state module with no LLM
dependency.

It implements:

- finite complex amplitudes;
- verifier-informed phase-like marking;
- a Hermitian generator;
- bounded Cayley evolution;
- normalized probability-like measurements;
- an explicit disabled/identity path;
- numerical and gradient diagnostics.

It does not implement physical quantum computation, enumerate all possible
solutions, or guarantee that interference removes incorrect reasoning.
`quality` is an external signal and must be validated independently.

Run the focused tests from the repository root:

```text
python -m unittest discover -s qreason/tests -p "test_*.py"
```

The LLM adapter, data evaluation and training loop are intentionally deferred
until these mathematical invariants and classical baselines are stable.

Run the synthetic mechanism check:

```text
python -m qreason.evaluate_synthetic
```

The synthetic oracle is intentionally known and is not evidence of language
reasoning improvement. It only verifies that a supplied quality signal can
affect finite hypothesis scores without numerical instability.

Run the blind sanity benchmark:

```text
python -m qreason.evaluate_blind
```

This benchmark deliberately supplies no quality labels to the core. Labels are
used only after inference to calculate accuracy against a chance reference.
An untrained QReason module is not expected to solve these tasks; this stage
checks for leakage, finite outputs and stable invariants before any learned
reasoning experiment.

The trainable synthetic calibration can be run with:

```text
python -m qreason.train_feature_task
```

This is not language-model training. It tests whether a QReason projection and
quality head can learn a fixed feature-to-hypothesis mapping, while the
real-valued control provides the required classical comparison.

The runner uses a shared hidden task rule with a held-out test split and reports
parameter counts. A result on this synthetic task is still only calibration
evidence, not proof of improved language reasoning.

The Boolean constraint benchmark can be run with:

```text
python -m qreason.evaluate_boolean
```

Each example contains candidate Boolean assignments and their clause
evaluations. The label is used only for loss and evaluation; the models receive
the candidate features, not an external validity score.

The leakage-free raw-clause benchmark can be run with:

```text
python -m qreason.evaluate_raw_boolean
```

It supplies raw signed clauses and candidate assignments only; clause
satisfaction is computed only to create labels and is not included in the
model input.

The parameter-matched five-seed version can be run with:

```text
python -m qreason.evaluate_raw_multi_seed
```

The guaranteed-unique Boolean benchmark can be run with:

```text
python -m qreason.evaluate_unique_boolean
```

Its generator constructs one satisfying candidate and a dedicated violated
clause for every distractor, then verifies the unique-solution invariant before
training.

The harder near-miss benchmark increases clause/variable complexity and
parameter-matches the feed-forward control to the QReason model:

```text
python -m qreason.evaluate_hard_boolean
```

The structured-control benchmark adds a shared candidate-clause interaction
encoder and compares a feed-forward control, a recurrent real-valued control,
and QReason under the same encoded inputs:

```text
python -m qreason.evaluate_structured
```

Run the five-seed comparison:

```text
python -m qreason.evaluate_multi_seed
```

The classical control uses the closest practical hidden width and reports its
parameter-count difference explicitly; this prevents an unqualified
superiority claim when QReason has more capacity.
