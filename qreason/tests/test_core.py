import unittest

import torch

from qreason import QReasonConfig, QReasonCore


class QReasonCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.core = QReasonCore(QReasonConfig(state_dim=8, cycles=3))
        self.amplitudes = torch.randn(4, 8)
        self.quality = torch.tensor(
            [[1.0, -1.0, 0.0, 0.2, -0.2, 0.0, 0.4, -0.4]] * 4
        )

    def test_probabilities_are_finite_and_normalized(self) -> None:
        output = self.core(self.amplitudes, self.quality)
        self.assertTrue(torch.isfinite(output.probabilities).all())
        self.assertTrue(torch.allclose(
            output.probabilities.sum(-1),
            torch.ones(4),
            atol=1e-5,
        ))

    def test_cayley_operator_is_unitary(self) -> None:
        operator = self.core.unitary()
        identity = torch.eye(8, dtype=operator.dtype)
        error = torch.linalg.matrix_norm(operator.conj().T @ operator - identity)
        self.assertLess(float(error.detach()), 1e-5)

    def test_state_norm_is_preserved(self) -> None:
        output = self.core(self.amplitudes, self.quality)
        self.assertLess(float(output.norm_error.detach()), 1e-5)

    def test_disabled_mode_is_identity_on_normalized_state(self) -> None:
        output = self.core(self.amplitudes, self.quality, enabled=False)
        expected = self.amplitudes.to(torch.complex64)
        expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(output.amplitudes, expected, atol=1e-6))

    def test_quality_bias_ranks_positive_quality_higher(self) -> None:
        output = self.core(torch.ones(1, 8), self.quality[:1])
        self.assertGreater(
            float(output.scores[0, 0].detach()),
            float(output.scores[0, 1].detach()),
        )

    def test_gradients_reach_generator_and_phase(self) -> None:
        output = self.core(self.amplitudes, self.quality)
        output.scores.mean().backward()
        self.assertIsNotNone(self.core.generator_factor.grad)
        self.assertIsNotNone(self.core.phase_scale.grad)
        self.assertTrue(torch.isfinite(self.core.generator_factor.grad).all())


if __name__ == "__main__":
    unittest.main()
