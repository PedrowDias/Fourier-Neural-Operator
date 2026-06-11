import torch
import torch.nn as nn
import torch.nn.functional as F

from fno.models.base import Operator


class SpectralConv1d(nn.Module):
    '''Spectral convolution layer for 1D inputs.

    Implements the core FNO operation:
        1. FFT the input into frequency space
        2. Multiply the lowest n_modes Fourier coefficients by learned weights
        3. IFFT back to physical space

    The weights are complex-valued and learned end-to-end. Only the lowest
    n_modes frequencies are kept — higher frequencies are zeroed out. This
    acts as a learned low-pass filter and is what makes the operation
    resolution-invariant: the weights don't depend on the grid size.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        n_modes:      Number of Fourier modes to keep. Must be <= n_grid // 2.
    '''

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes, 2)
        )

    def _complex_weights(self) -> torch.Tensor:
        return torch.view_as_complex(self.weights.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply spectral convolution.

        Args:
            x: Shape (batch, in_channels, n_grid). n_grid can be any size.

        Returns:
            Shape (batch, out_channels, n_grid).
        '''
        batch, _, n_grid = x.shape

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch, self.out_channels, n_grid // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.n_modes] = torch.einsum(
            'bix,iox->box',
            x_ft[:, :, :self.n_modes],
            self._complex_weights(),
        )

        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FNOBlock1d(nn.Module):
    '''One FNO layer: spectral conv + linear bypass + activation.

    Args:
        channels: Number of channels (same in and out).
        n_modes:  Number of Fourier modes for the spectral convolution.
    '''

    def __init__(self, channels: int, n_modes: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, n_modes)
        self.bypass        = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.bypass(x))


class FNO1d(Operator):
    '''Fourier Neural Operator for 1D parametric PDEs.

    Architecture (following Li et al. 2020):
        1. Input projection:  lifts (batch, n_grid) → (batch, channels, n_grid)
        2. FNO blocks:        n_layers stacked FNOBlock1d layers
        3. Output projection: projects (batch, channels, n_grid) → (batch, n_grid)

    The spatial coordinate x is appended as an extra input channel before
    the lifting layer. Crucially, this grid is computed dynamically from the
    actual input size at each forward pass — not stored as a fixed buffer.
    This is what enables true resolution invariance: the same trained model
    handles any grid size without modification.

    Args:
        channels: Width of the hidden representation.
        n_modes:  Number of Fourier modes to keep per layer.
        n_layers: Number of FNO blocks.
    '''

    def __init__(
        self,
        channels: int = 64,
        n_modes:  int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        # +1 for the appended spatial coordinate channel
        self.input_projection  = nn.Conv1d(2, channels, kernel_size=1)
        self.fno_blocks        = nn.Sequential(*[FNOBlock1d(channels, n_modes) for _ in range(n_layers)])
        self.output_projection = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Map a batch of initial conditions to predicted solutions.

        Args:
            x: Shape (batch, n_grid). n_grid can be any size at inference.

        Returns:
            Shape (batch, n_grid).
        '''
        batch, n_grid = x.shape

        # Build spatial grid dynamically from actual input size
        grid = torch.linspace(0, 1, n_grid, device=x.device).reshape(1, 1, n_grid)
        grid = grid.expand(batch, -1, -1)          # (batch, 1, n_grid)

        x = x.unsqueeze(1)                         # (batch, 1, n_grid)
        x = torch.cat([x, grid], dim=1)            # (batch, 2, n_grid)

        x = self.input_projection(x)               # (batch, channels, n_grid)
        x = self.fno_blocks(x)                     # (batch, channels, n_grid)
        x = self.output_projection(x)              # (batch, 1, n_grid)

        return x.squeeze(1)                        # (batch, n_grid)
