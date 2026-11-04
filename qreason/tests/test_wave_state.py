import unittest

import torch
from torch import nn

from qreason.model_integration import attach_qreason
from qreason.wave_state import QReasonAdapter, WaveStateConfig, WaveStateEngine


class WaveStateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.config = WaveStateConfig(hidden_dim=6, hypotheses=5, cycles=2)
        self.hidden = torch.randn(3, 4, 6)
        self.engine = WaveStateEngine(self.config)

    def test_creates_multiple_complex_hypotheses(self) -> None:
        output = self.engine(self.hidden)
        self.assertEqual(output.state.shape, (3, 4, 5, 6))
        self.assertTrue(torch.is_complex(output.state))
        self.assertEqual(output.probabilities.shape, (3, 4, 5))

    def test_measurement_is_normalized_and_finite(self) -> None:
        output = self.engine(self.hidden)
        self.assertTrue(torch.isfinite(output.probabilities).all())
        self.assertTrue(torch.allclose(
            output.probabilities.sum(-1),
            torch.ones(3, 4),
            atol=1e-5,
        ))
        self.assertLess(float(output.norm_error.detach()), 1e-5)
        self.assertLess(float(output.unitary_error.detach()), 1e-5)

    def test_input_phase_changes_measured_state(self) -> None:
        output = self.engine(self.hidden)
        with torch.no_grad():
            self.engine.phase_scale.fill_(1.0)
        changed = self.engine(self.hidden)
        self.assertFalse(torch.allclose(
            output.probabilities, changed.probabilities, atol=1e-6
        ))

    def test_disabled_adapter_is_exact_identity(self) -> None:
        adapter = QReasonAdapter(self.config)
        result, output = adapter(self.hidden, enabled=False)
        self.assertTrue(torch.equal(result, self.hidden))
        self.assertTrue(torch.equal(output.collapsed, self.hidden.reshape(-1, 6)))

    def test_adapter_preserves_shape_and_has_gradients(self) -> None:
        adapter = QReasonAdapter(self.config)
        with torch.no_grad():
            adapter.residual_gate.fill_(0.2)
        result, _ = adapter(self.hidden)
        self.assertEqual(result.shape, self.hidden.shape)
        result.square().mean().backward()
        self.assertIsNotNone(adapter.engine.hypothesis.weight.grad)
        self.assertTrue(torch.isfinite(adapter.engine.hypothesis.weight.grad).all())

    def test_coherent_opposite_hypotheses_cancel(self) -> None:
        state = torch.zeros(1, 5, 6, dtype=torch.complex64)
        state[0, 0, 0] = 1.0
        state[0, 1, 0] = -1.0
        collapsed, interference_norm = self.engine.coherent_collapse(state)
        self.assertEqual(float(interference_norm), 0.0)
        self.assertEqual(float(torch.linalg.vector_norm(collapsed)), 0.0)

    def test_hook_transforms_tensor_and_tuple_hidden_outputs(self) -> None:
        class Block(nn.Module):
            def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
                return hidden * 2.0, "cache"

        block = Block()
        hook = attach_qreason(block, self.config, enabled=False)
        hidden = torch.randn(2, 3, 6)
        result = block(hidden)
        self.assertEqual(result[0].shape, hidden.shape)
        self.assertEqual(result[1], "cache")
        self.assertTrue(torch.equal(result[0], hidden * 2.0))
        hook.remove()


if __name__ == "__main__":
    unittest.main()
