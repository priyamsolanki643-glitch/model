import unittest

import torch

from qreason.core import QReasonConfig
from qreason.raw_boolean import (
    RawQReasonClassifier,
    RawRealClassifier,
    evaluate_raw,
    make_raw_boolean_task,
)


class RawBooleanTests(unittest.TestCase):
    def test_features_do_not_include_precomputed_satisfaction(self) -> None:
        task = make_raw_boolean_task(
            samples=12, candidate_count=6, variable_count=4, clause_count=5, seed=31
        )
        self.assertEqual(task.candidate_features.shape, (12, 6, 4 + 15))
        self.assertTrue(torch.equal(task.labels, task.labels.long()))

    def test_raw_models_produce_finite_scores(self) -> None:
        task = make_raw_boolean_task(samples=8, candidate_count=6, seed=32)
        config = QReasonConfig(state_dim=6, cycles=1)
        feature_dim = task.candidate_features.shape[-1]
        qreason = RawQReasonClassifier(feature_dim, config)
        real = RawRealClassifier(feature_dim, 6)
        self.assertEqual(evaluate_raw(qreason, task)["finite"], 1.0)
        self.assertEqual(evaluate_raw(real, task)["finite"], 1.0)


if __name__ == "__main__":
    unittest.main()
