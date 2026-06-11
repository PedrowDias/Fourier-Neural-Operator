import torch

from fno.training.metrics import absolute_l2_error, relative_l2_error


class TestRelativeL2Error:

    def test_perfect_prediction_is_zero(self):
        x = torch.randn(8, 64)
        assert relative_l2_error(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_zero_prediction(self):
        target = torch.ones(4, 32)
        pred = torch.zeros(4, 32)
        # error norm == target norm for each sample, so relative error == 1.0
        assert relative_l2_error(pred, target).item() == pytest.approx(1.0, rel=1e-4)

    def test_output_is_scalar(self):
        pred = torch.randn(16, 64)
        target = torch.randn(16, 64)
        result = relative_l2_error(pred, target)
        assert result.shape == torch.Size([])


import pytest  # noqa: E402 — kept at bottom to mirror project import style
