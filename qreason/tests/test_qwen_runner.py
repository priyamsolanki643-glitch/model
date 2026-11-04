import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from qreason.qwen_kaggle import (
    _encode,
    attach_layers,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class _Config:
    hidden_size = 6


class _Host(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])
        self.config = _Config()


class _Tokenizer:
    def __call__(self, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.ones(1, len(text), dtype=torch.long)}


class QwenRunnerTests(unittest.TestCase):
    def test_attach_layers_uses_host_hidden_width(self) -> None:
        host = _Host()
        hooks = attach_layers(host, layer_indices=[0], qubits=3, enabled=False)
        hidden = torch.randn(1, 4, 6)
        result = host.model.layers[0](hidden)
        self.assertTrue(torch.equal(result, hidden))
        hooks[0].remove()

    def test_encoder_keeps_tensor_device(self) -> None:
        inputs = _encode(_Tokenizer(), "abc", torch.device("cpu"))
        self.assertEqual(inputs["input_ids"].shape, (1, 3))

    def test_adapter_checkpoint_round_trip(self) -> None:
        host = _Host()
        hooks = attach_layers(host, layer_indices=[0], qubits=3, enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter_checkpoint(
                hooks, path, model_id="test", layer_indices=[0], qubits=3
            )
            checkpoint = load_adapter_checkpoint(hooks, path)
        self.assertEqual(checkpoint["model"], "test")
        hooks[0].remove()


if __name__ == "__main__":
    unittest.main()
