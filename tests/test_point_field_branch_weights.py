from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from ai_model.model.point_field import (
    POINT_FIELD_CHECKPOINT_VERSION,
    SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS,
    DirectPointFieldModel,
    load_compatible_point_field_state,
)


class _ConstantBranch(nn.Module):
    def __init__(self, values: tuple[float, ...]):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(waveform.shape[0], -1)


class PointFieldBranchWeightTests(unittest.TestCase):
    def test_cnn_and_lstm_weights_change_forward_result(self):
        model = DirectPointFieldModel(
            point_count=1,
            hidden_dim=2,
            latent_dim=2,
            chunk_size=1,
            fixed_weight_cnn=2.0,
            fixed_weight_lstm=3.0,
        )
        model.encoder = _ConstantBranch((4.0, 5.0))
        model.lstm_encoder = _ConstantBranch((7.0, 11.0))
        model.backbone = nn.Identity()
        model.heads = nn.ModuleList([nn.Linear(4, 1, bias=False)])
        with torch.no_grad():
            model.heads[0].weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))

        waveform = torch.zeros((2, 1, 8), dtype=torch.float32)
        first = model(waveform)
        torch.testing.assert_close(first, torch.full((2, 1), 29.0))

        with torch.no_grad():
            model.weight_cnn.fill_(5.0)
            model.weight_lstm.fill_(7.0)
        second = model(waveform)
        torch.testing.assert_close(second, torch.full((2, 1), 69.0))

    def test_learnable_branch_weights_receive_gradients(self):
        model = DirectPointFieldModel(
            point_count=1,
            hidden_dim=2,
            latent_dim=2,
            chunk_size=1,
            learnable_branch_weights=True,
            fixed_weight_cnn=2.0,
            fixed_weight_lstm=3.0,
        )
        model.encoder = _ConstantBranch((4.0, 5.0))
        model.lstm_encoder = _ConstantBranch((7.0, 11.0))
        model.backbone = nn.Identity()
        model.heads = nn.ModuleList([nn.Linear(4, 1, bias=False)])
        with torch.no_grad():
            model.heads[0].weight.copy_(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))

        model(torch.zeros((2, 1, 8), dtype=torch.float32)).sum().backward()

        self.assertIsInstance(model.weight_cnn, nn.Parameter)
        self.assertIsInstance(model.weight_lstm, nn.Parameter)
        torch.testing.assert_close(model.weight_cnn.grad, torch.tensor([8.0]))
        torch.testing.assert_close(model.weight_lstm.grad, torch.tensor([14.0]))

    def test_legacy_state_without_branch_weights_loads_as_unity(self):
        torch.manual_seed(17)
        legacy_source = DirectPointFieldModel(
            point_count=3,
            hidden_dim=4,
            latent_dim=4,
            chunk_size=2,
            fixed_weight_cnn=1.0,
            fixed_weight_lstm=1.0,
        ).eval()
        legacy_state = {
            key: value.clone()
            for key, value in legacy_source.state_dict().items()
            if key not in {"weight_cnn", "weight_lstm"}
        }
        restored = DirectPointFieldModel(
            point_count=3,
            hidden_dim=4,
            latent_dim=4,
            chunk_size=2,
            fixed_weight_cnn=0.25,
            fixed_weight_lstm=0.50,
        ).eval()

        missing = load_compatible_point_field_state(restored, legacy_state)

        self.assertEqual(set(missing), {"weight_cnn", "weight_lstm"})
        torch.testing.assert_close(restored.weight_cnn, torch.ones_like(restored.weight_cnn))
        torch.testing.assert_close(restored.weight_lstm, torch.ones_like(restored.weight_lstm))
        waveform = torch.randn((2, 1, 16), dtype=torch.float32)
        with torch.no_grad():
            torch.testing.assert_close(restored(waveform), legacy_source(waveform))

    def test_compatibility_loader_rejects_unrelated_state_mismatch(self):
        model = DirectPointFieldModel(2, hidden_dim=4, latent_dim=4, chunk_size=2)
        state = dict(model.state_dict())
        state.pop("backbone.0.weight")
        with self.assertRaisesRegex(RuntimeError, "backbone.0.weight"):
            load_compatible_point_field_state(model, state)

        state = dict(model.state_dict())
        state["unexpected.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected.weight"):
            load_compatible_point_field_state(model, state)

    def test_current_checkpoint_version_retains_legacy_reader(self):
        self.assertEqual(POINT_FIELD_CHECKPOINT_VERSION, 3)
        self.assertEqual(SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS, (2, 3))


if __name__ == "__main__":
    unittest.main()
