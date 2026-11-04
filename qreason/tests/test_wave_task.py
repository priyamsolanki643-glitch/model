import unittest

import torch

from qreason.wave_task import (
    WaveClassifier,
    evaluate_wave_model,
    make_wave_task_split,
)
from qreason.trainable import parameter_count


class WaveTaskTests(unittest.TestCase):
    def test_split_is_reproducible_and_labels_are_hidden(self) -> None:
        first = make_wave_task_split(train_samples=32, test_samples=16, seed=4)
        second = make_wave_task_split(train_samples=32, test_samples=16, seed=4)
        self.assertTrue(torch.equal(first.train.features, second.train.features))
        self.assertTrue(torch.equal(first.test.labels, second.test.labels))
        self.assertEqual(first.train.features.shape[-1], 12)

    def test_wave_classifier_outputs_finite_hypothesis_scores(self) -> None:
        task = make_wave_task_split(train_samples=16, test_samples=8, seed=8).test
        model = WaveClassifier(12, 8)
        metrics = evaluate_wave_model(model, task)
        self.assertEqual(parameter_count(model) > 0, True)
        self.assertEqual(metrics["finite"], 1.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
