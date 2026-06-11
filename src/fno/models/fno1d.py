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

        # Complex weights, stored as real tensor of shape (in, out, n_modes, 2)
        # where the last dim is [real, imag]. This avoids dtype issues across
        # PyTorch versions.
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes, 2)
        )

    def _complex_weights(self) -> torch.Tensor:
        return torch.view_as_complex(self.weights.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply spectral convolution.

        Args:
            x: Shape (batch, in_channels, n_grid).

        Returns:
            Shape (batch, out_channels, n_grid).
        '''
        batch, _, n_grid = x.shape

        # FFT along the spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, n_grid//2 + 1)

        # Multiply lowest n_modes by learned weights
        out_ft = torch.zeros(
            batch, self.out_channels, n_grid // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.n_modes] = torch.einsum(
            'bix,iox->box',
            x_ft[:, :, :self.n_modes],
            self._complex_weights(),
        )

        # IFFT back to physical space
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)  # (batch, out_channels, n_grid)


class FNOBlock1d(nn.Module):
    '''One FNO layer: spectral conv + linear bypass + activation.

    The bypass (W) is a pointwise linear transform applied in physical space.
    Adding it to the spectral conv output before the activation mirrors the
    residual structure in ResNets and stabilises training.

    Args:
        channels: Number of channels (same in and out, enabling residual stacking).
        n_modes:  Number of Fourier modes for the spectral convolution.
    '''

    def __init__(self, channels: int, n_modes: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, n_modes)
        self.bypass        = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Shape (batch, channels, n_grid).

        Returns:
            Shape (batch, channels, n_grid).
        '''
        return F.gelu(self.spectral_conv(x) + self.bypass(x))


class FNO1d(Operator):
    '''Fourier Neural Operator for 1D parametric PDEs.

    Architecture (following Li et al. 2020):
        1. Input projection:  lifts (batch, n_grid) → (batch, channels, n_grid)
        2. FNO blocks:        n_layers stacked FNOBlock1d layers
        3. Output projection: projects (batch, channels, n_grid) → (batch, n_grid)

    The spatial coordinate x is appended as an extra input channel before
    the lifting layer. This gives the network positional information, which
    improves accuracy on non-translation-invariant problems.

    Args:
        n_grid:   Number of spatial grid points.
        channels: Width of the hidden representation (lifted space).
        n_modes:  Number of Fourier modes to keep per layer.
        n_layers: Number of FNO blocks.
    '''

    def __init__(
        self,
        n_grid:   int = 64,
        channels: int = 64,
        n_modes:  int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.n_grid = n_grid

        # +1 because we append the spatial coordinate as an extra channel
        self.input_projection  = nn.Conv1d(2, channels, kernel_size=1)
        self.fno_blocks        = nn.Sequential(*[FNOBlock1d(channels, n_modes) for _ in range(n_layers)])
        self.output_projection = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

        # Fixed spatial grid, registered as a buffer so it moves to the
        # correct device automatically when model.to(device) is called
        x = torch.linspace(0, 1, n_grid).reshape(1, 1, n_grid)
        self.register_buffer('x_grid', x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Map a batch of initial conditions to predicted solutions.

        Args:
            x: Shape (batch, n_grid).

        Returns:
            Shape (batch, n_grid).
        '''
        batch = x.shape[0]

        # Reshape to (batch, 1, n_grid) and append spatial coordinate
        x = x.unsqueeze(1)
        grid = self.x_grid.expand(batch, -1, -1)   # (batch, 1, n_grid)
        x = torch.cat([x, grid], dim=1)             # (batch, 2, n_grid)

        x = self.input_projection(x)                # (batch, channels, n_grid)
        x = self.fno_blocks(x)                      # (batch, channels, n_grid)
        x = self.output_projection(x)               # (batch, 1, n_grid)

        return x.squeeze(1)                         # (batch, n_grid)
