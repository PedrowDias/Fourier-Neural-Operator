import torch
import pytest

from fno.models.fno1d import FNO1d, FNOBlock1d, SpectralConv1d


class TestSpectralConv1d:

    def test_output_shape(self):
        layer = SpectralConv1d(in_channels=4, out_channels=8, n_modes=12)
        x = torch.randn(6, 4, 64)
        assert layer(x).shape == (6, 8, 64)

    def test_gradients_flow(self):
        layer = SpectralConv1d(in_channels=4, out_channels=4, n_modes=12)
        x = torch.randn(4, 4, 64)
        layer(x).sum().backward()
        assert layer.weights.grad is not None

    def test_output_is_real(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, n_modes=8)
        x = torch.randn(4, 2, 64)
        assert layer(x).is_floating_point()


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
        model = FNO1d()
        x = torch.randn(8, 64)
        assert model(x).shape == (8, 64)

    def test_different_batch_sizes(self):
        model = FNO1d()
        for batch in [1, 4, 16]:
            assert model(torch.randn(batch, 64)).shape == (batch, 64)

    def test_parameter_count_is_positive(self):
        assert FNO1d().parameter_count() > 0

    def test_gradients_flow(self):
        model = FNO1d()
        model(torch.randn(4, 64)).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_resolution_invariance_32(self):
        model = FNO1d()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 32))
        assert out.shape == (4, 32)

    def test_resolution_invariance_128(self):
        model = FNO1d()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 128))
        assert out.shape == (4, 128)

    def test_resolution_invariance_256(self):
        model = FNO1d()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 256))
        assert out.shape == (4, 256)

    def test_same_weights_different_resolutions(self):
        '''The same model instance should run at any resolution without modification.'''
        model = FNO1d()
        model.eval()
        with torch.no_grad():
            out_64  = model(torch.randn(2, 64))
            out_128 = model(torch.randn(2, 128))
        assert out_64.shape  == (2, 64)
        assert out_128.shape == (2, 128)
