import unittest

import torch

from qreason.core import QReasonConfig
from qreason.structured_boolean import (
    StructuredQReason,
    encode_candidate_clause_interactions,
)
from qreason.unique_boolean import make_unique_boolean_task


class StructuredBooleanTests(unittest.TestCase):
    def test_encoder_has_expected_shape_and_is_finite(self) -> None:
        task = make_unique_boolean_task(
            samples=12, candidate_count=6, variable_count=5, seed=91
        )
        encoded = encode_candidate_clause_interactions(task)
        self.assertEqual(encoded.features.shape, (12, 6, 5 + 5 * 3 + 1))
        self.assertTrue(torch.isfinite(encoded.features).all())

    def test_qreason_control_scores_candidates(self) -> None:
        task = make_unique_boolean_task(samples=8, candidate_count=6, seed=92)
        encoded = encode_candidate_clause_interactions(task)
        model = StructuredQReason(
            encoded.features.shape[-1],
            QReasonConfig(state_dim=6, cycles=1),
        )
        self.assertEqual(model(encoded.features).shape, (8, 6))


if __name__ == "__main__":
    unittest.main()
