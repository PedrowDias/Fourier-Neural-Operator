import torch
import pytest

from fno.models.fno2d import FNO2d, FNOBlock2d, SpectralConv2d


class TestSpectralConv2d:

    def test_output_shape(self):
        layer = SpectralConv2d(4, 8, n_modes_x=8, n_modes_y=8)
        x = torch.randn(4, 4, 32, 32)
        assert layer(x).shape == (4, 8, 32, 32)

    def test_gradients_flow(self):
        layer = SpectralConv2d(4, 4, n_modes_x=8, n_modes_y=8)
        x = torch.randn(2, 4, 32, 32)
        layer(x).sum().backward()
        assert layer.weights.grad is not None

    def test_output_is_real(self):
        layer = SpectralConv2d(2, 2, n_modes_x=8, n_modes_y=8)
        x = torch.randn(2, 2, 32, 32)
        assert layer(x).is_floating_point()


class TestFNOBlock2d:

    def test_output_shape(self):
        block = FNOBlock2d(channels=16, n_modes_x=8, n_modes_y=8)
        x = torch.randn(2, 16, 32, 32)
        assert block(x).shape == (2, 16, 32, 32)


class TestFNO2d:

    def test_output_shape(self):
        model = FNO2d()
        x = torch.randn(2, 64, 64)
        assert model(x).shape == (2, 64, 64)

    def test_different_batch_sizes(self):
        model = FNO2d()
        for batch in [1, 2, 4]:
            assert model(torch.randn(batch, 64, 64)).shape == (batch, 64, 64)

    def test_parameter_count_is_positive(self):
        assert FNO2d().parameter_count() > 0

    def test_gradients_flow(self):
        model = FNO2d()
        model(torch.randn(2, 32, 32)).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_resolution_invariance_32(self):
        model = FNO2d()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 32, 32))
        assert out.shape == (2, 32, 32)

    def test_resolution_invariance_128(self):
        model = FNO2d()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 128, 128))
        assert out.shape == (2, 128, 128)

    def test_same_weights_different_resolutions(self):
        model = FNO2d()
        model.eval()
        with torch.no_grad():
            out_32  = model(torch.randn(1, 32, 32))
            out_64  = model(torch.randn(1, 64, 64))
        assert out_32.shape  == (1, 32, 32)
        assert out_64.shape  == (1, 64, 64)
