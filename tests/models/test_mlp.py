import torch
import pytest

from fno.models.mlp import MLP


class TestMLP:

    def test_output_shape(self):
        model = MLP(n_grid=64)
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 64)

    def test_different_batch_sizes(self):
        model = MLP(n_grid=64)
        for batch in [1, 4, 32]:
            out = model(torch.randn(batch, 64))
            assert out.shape == (batch, 64)

    def test_parameter_count_is_positive(self):
        model = MLP(n_grid=64)
        assert model.parameter_count() > 0

    def test_invalid_n_layers_raises(self):
        with pytest.raises(ValueError, match='n_layers must be >= 2'):
            MLP(n_grid=64, n_layers=1)

    def test_output_changes_with_input(self):
        model = MLP(n_grid=64)
        x1 = torch.randn(4, 64)
        x2 = torch.randn(4, 64)
        assert not torch.allclose(model(x1), model(x2))

    def test_gradients_flow(self):
        model = MLP(n_grid=64)
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
