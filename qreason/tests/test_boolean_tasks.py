import unittest

import torch

from qreason.boolean_tasks import (
    CandidateQReasonClassifier,
    CandidateRealClassifier,
    evaluate_boolean,
    make_boolean_task,
)
from qreason.core import QReasonConfig


class BooleanTaskTests(unittest.TestCase):
    def test_task_is_reproducible_and_has_valid_shapes(self) -> None:
        first = make_boolean_task(samples=24, candidate_count=6, seed=21)
        second = make_boolean_task(samples=24, candidate_count=6, seed=21)
        self.assertTrue(torch.equal(first.candidates, second.candidates))
        self.assertTrue(torch.equal(first.labels, second.labels))
        self.assertEqual(first.candidates.shape, (24, 6, 14))
        self.assertTrue((first.labels >= 0).all())
        self.assertTrue((first.labels < 6).all())

    def test_models_produce_finite_candidate_scores(self) -> None:
        task = make_boolean_task(samples=12, candidate_count=6, seed=22)
        config = QReasonConfig(state_dim=6, cycles=1)
        qreason = CandidateQReasonClassifier(task.candidates.shape[-1], config)
        real = CandidateRealClassifier(task.candidates.shape[-1], 6)
        self.assertEqual(qreason(task.candidates).shape, (12, 6))
        self.assertEqual(real(task.candidates).shape, (12, 6))
        self.assertEqual(evaluate_boolean(qreason, task)["finite"], 1.0)
        self.assertEqual(evaluate_boolean(real, task)["finite"], 1.0)


if __name__ == "__main__":
    unittest.main()
