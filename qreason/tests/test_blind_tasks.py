import unittest

import torch

from qreason import QReasonConfig, QReasonCore
from qreason.blind_tasks import evaluate_blind_tasks, make_blind_tasks


class BlindTaskTests(unittest.TestCase):
    def test_blind_evaluation_does_not_pass_labels_to_core(self) -> None:
        tasks = make_blind_tasks(batch_size=64, state_dim=8, seed=10)
        core = QReasonCore(QReasonConfig(state_dim=8, cycles=2))
        first = [
            core(task.amplitudes, torch.zeros_like(task.amplitudes)).scores
            for task in tasks
        ]
        for task in tasks:
            task.labels.fill_(0)
        second = [
            core(task.amplitudes, torch.zeros_like(task.amplitudes)).scores
            for task in tasks
        ]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_blind_tasks_are_finite_and_have_chance_reference(self) -> None:
        tasks = make_blind_tasks(batch_size=32, state_dim=8, seed=11)
        core = QReasonCore(QReasonConfig(state_dim=8, cycles=2))
        results = evaluate_blind_tasks(core, tasks)
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(result["chance_top1"], 0.125)
            self.assertEqual(result["finite"], 1.0)
            self.assertLess(result["unitary_error"], 1e-5)
            self.assertLess(result["norm_error"], 1e-5)

    def test_blind_task_generation_is_reproducible(self) -> None:
        first = make_blind_tasks(batch_size=12, state_dim=6, seed=12)
        second = make_blind_tasks(batch_size=12, state_dim=6, seed=12)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left.amplitudes, right.amplitudes))
            self.assertTrue(torch.equal(left.labels, right.labels))


if __name__ == "__main__":
    unittest.main()
