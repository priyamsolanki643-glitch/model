import unittest

import torch
from torch import nn

from qreason.model_integration import attach_statevector_qreason
from qreason.statevector import StateVectorConfig, StateVectorEngine


class StateVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.config = StateVectorConfig(hidden_dim=6, qubits=4, cycles=2)
        self.engine = StateVectorEngine(self.config)
        self.hidden = torch.randn(3, 2, 6)

    def test_register_has_exponential_basis_dimension(self) -> None:
        output = self.engine(self.hidden)
        self.assertEqual(output.amplitudes.shape, (3, 2, 16))
        self.assertEqual(output.probabilities.shape, (3, 2, 16))
        self.assertTrue(torch.is_complex(output.amplitudes))

    def test_probabilities_are_normalized(self) -> None:
        output = self.engine(self.hidden)
        self.assertTrue(torch.allclose(
            output.probabilities.sum(-1),
            torch.ones(3, 2),
            atol=1e-5,
        ))
        self.assertLess(float(output.norm_error.detach()), 1e-5)

    def test_hadamard_interference_is_unitary(self) -> None:
        state = torch.randn(4, 16, dtype=torch.complex64)
        before = torch.linalg.vector_norm(state, dim=-1)
        from qreason.statevector import _hadamard_transform
        after = torch.linalg.vector_norm(_hadamard_transform(state), dim=-1)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_disabled_register_is_deterministic_basis_state(self) -> None:
        output = self.engine(self.hidden, enabled=False)
        self.assertTrue(torch.equal(output.probabilities[..., 0], torch.ones(3, 2)))
        self.assertTrue(torch.equal(output.probabilities[..., 1:], torch.zeros(3, 2, 15)))

    def test_phase_oracle_receives_gradients(self) -> None:
        output = self.engine(self.hidden)
        output.measured_scores.mean().backward()
        self.assertIsNotNone(self.engine.oracle.weight.grad)
        self.assertTrue(torch.isfinite(self.engine.oracle.weight.grad).all())

    def test_statevector_hook_preserves_hidden_shape(self) -> None:
        class Block(nn.Module):
            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                return hidden

        block = Block()
        hook = attach_statevector_qreason(block, self.config, enabled=False)
        output = block(self.hidden)
        self.assertEqual(output.shape, self.hidden.shape)
        self.assertTrue(torch.equal(output, self.hidden))
        hook.remove()


if __name__ == "__main__":
    unittest.main()
