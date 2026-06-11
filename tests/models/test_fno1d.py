import torch
import pytest

from fno.models.fno1d import FNO1d, FNOBlock1d, SpectralConv1d


class TestSpectralConv1d:

    def test_output_shape(self):
        layer = SpectralConv1d(in_channels=4, out_channels=8, n_modes=12)
        x = torch.randn(6, 4, 64)
        out = layer(x)
        assert out.shape == (6, 8, 64)

    def test_gradients_flow(self):
        layer = SpectralConv1d(in_channels=4, out_channels=4, n_modes=12)
        x = torch.randn(4, 4, 64)
        layer(x).sum().backward()
        assert layer.weights.grad is not None

    def test_output_is_real(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, n_modes=8)
        x = torch.randn(4, 2, 64)
        out = layer(x)
        assert out.is_floating_point()


class TestFNOBlock1d:

    def test_output_shape(self):
        block = FNOBlock1d(channels=32, n_modes=12)
        x = torch.randn(4, 32, 64)
        assert block(x).shape == (4, 32, 64)

    def test_residual_changes_input(self):
        block = FNOBlock1d(channels=32, n_modes=12)
        x = torch.randn(4, 32, 64)
        assert not torch.allclose(block(x), x)


class TestFNO1d:

    def test_output_shape(self):
        model = FNO1d(n_grid=64)
        x = torch.randn(8, 64)
        assert model(x).shape == (8, 64)

    def test_different_batch_sizes(self):
        model = FNO1d(n_grid=64)
        for batch in [1, 4, 16]:
            assert model(torch.randn(batch, 64)).shape == (batch, 64)

    def test_parameter_count_is_positive(self):
        model = FNO1d(n_grid=64)
        assert model.parameter_count() > 0

    def test_gradients_flow(self):
        model = FNO1d(n_grid=64)
        x = torch.randn(4, 64)
        model(x).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_resolution_invariance(self):
        '''Model trained at n_grid=64 should run at n_grid=128 without retraining.

        This is the defining property of neural operators. We just check that
        the forward pass doesn't crash at a different resolution — accuracy
        evaluation happens in the benchmark script.
        '''
        model = FNO1d(n_grid=64)
        model.eval()
        x_high_res = torch.randn(4, 128)
        # Manually update the grid buffer to match new resolution
        model.x_grid = torch.linspace(0, 1, 128).reshape(1, 1, 128)
        out = model(x_high_res)
        assert out.shape == (4, 128)
