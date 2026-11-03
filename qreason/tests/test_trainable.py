import unittest

import torch

from qreason.core import QReasonConfig
from qreason.trainable import (
    QReasonClassifier,
    evaluate_classifier,
    make_feature_split,
    make_feature_task,
    parameter_count,
    train_classifier,
)


class TrainableQReasonTests(unittest.TestCase):
    def test_feature_task_is_reproducible(self) -> None:
        first = make_feature_task(samples=16, feature_dim=5, state_dim=4, seed=8)
        second = make_feature_task(samples=16, feature_dim=5, state_dim=4, seed=8)
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.labels, second.labels))

    def test_qreason_classifier_learns_a_small_task(self) -> None:
        torch.manual_seed(8)
        task = make_feature_task(samples=96, feature_dim=6, state_dim=4, seed=8)
        model = QReasonClassifier(
            feature_dim=6,
            config=QReasonConfig(state_dim=4, cycles=1, quality_strength=1.0),
        )
        before = evaluate_classifier(model, task)
        losses = train_classifier(model, task, steps=120, learning_rate=5e-3)
        after = evaluate_classifier(model, task)
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(after["top1_accuracy"], before["top1_accuracy"])
        self.assertEqual(after["finite"], 1.0)

    def test_feature_split_shares_task_rule_without_sharing_examples(self) -> None:
        split = make_feature_split(
            train_samples=16,
            test_samples=8,
            feature_dim=5,
            state_dim=4,
            seed=9,
        )
        self.assertFalse(torch.equal(split.train.features, split.test.features[:16]))
        self.assertEqual(split.train.features.shape[-1], split.test.features.shape[-1])

    def test_parameter_count_is_positive(self) -> None:
        model = QReasonClassifier(6, QReasonConfig(state_dim=4))
        self.assertGreater(parameter_count(model), 0)


if __name__ == "__main__":
    unittest.main()
