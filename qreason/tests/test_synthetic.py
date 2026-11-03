import unittest

import torch

from qreason import QReasonConfig, QReasonCore
from qreason.synthetic import (
    evaluate_core,
    evaluate_uniform_baseline,
    make_constraint_batch,
)


class SyntheticEvaluationTests(unittest.TestCase):
    def test_constraint_batch_is_reproducible(self) -> None:
        first = make_constraint_batch(batch_size=8, state_dim=6, seed=3)
        second = make_constraint_batch(batch_size=8, state_dim=6, seed=3)
        self.assertTrue(torch.equal(first.amplitudes, second.amplitudes))
        self.assertTrue(torch.equal(first.quality, second.quality))
        self.assertTrue(torch.equal(first.valid_index, second.valid_index))

    def test_qreason_uses_quality_signal_to_rank_valid_state(self) -> None:
        batch = make_constraint_batch(batch_size=128, state_dim=8, seed=4)
        core = QReasonCore(QReasonConfig(state_dim=8, cycles=2, quality_strength=2.0))
        metrics = evaluate_core(core, batch)
        baseline = evaluate_uniform_baseline(batch)
        self.assertGreater(metrics["top1_accuracy"], baseline["top1_accuracy"])
        self.assertGreater(metrics["valid_probability"], baseline["valid_probability"])
        self.assertTrue(metrics["finite"])

    def test_disabled_core_does_not_claim_quality(self) -> None:
        batch = make_constraint_batch(batch_size=64, state_dim=8, seed=5)
        core = QReasonCore(QReasonConfig(state_dim=8, cycles=2, quality_strength=2.0))
        metrics = evaluate_core(core, batch, enabled=False)
        baseline = evaluate_uniform_baseline(batch)
        self.assertLessEqual(
            abs(metrics["valid_probability"] - baseline["valid_probability"]),
            0.05,
        )

    def test_invalid_probability_is_not_nan(self) -> None:
        batch = make_constraint_batch(batch_size=32, state_dim=10, seed=6)
        core = QReasonCore(QReasonConfig(state_dim=10, cycles=4))
        metrics = evaluate_core(core, batch)
        self.assertGreaterEqual(metrics["invalid_probability"], 0.0)
        self.assertLessEqual(metrics["invalid_probability"], 1.0)
        self.assertLess(metrics["unitary_error"], 1e-5)
        self.assertLess(metrics["norm_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
