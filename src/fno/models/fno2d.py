import torch
import torch.nn as nn
import torch.nn.functional as F

from fno.models.base import Operator


class SpectralConv2d(nn.Module):
    '''Spectral convolution layer for 2D inputs.

    Extends SpectralConv1d to two spatial dimensions. The operation:
        1. Apply 2D real FFT (rfft2) to get frequency representation
        2. Multiply the lowest (n_modes_x, n_modes_y) modes by learned weights
        3. Apply 2D inverse real FFT (irfft2) back to physical space

    For real inputs, rfft2 exploits Hermitian symmetry and returns an array
    of shape (..., nx, ny//2+1). We keep the top-left (n_modes_x, n_modes_y)
    block of modes, which corresponds to low frequencies in both directions.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        n_modes_x:    Fourier modes to keep in x direction.
        n_modes_y:    Fourier modes to keep in y direction.
    '''

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_modes_x:    int,
        n_modes_y:    int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes_x    = n_modes_x
        self.n_modes_y    = n_modes_y

        scale = 1.0 / (in_channels * out_channels)
        # Shape: (in, out, n_modes_x, n_modes_y, 2) — last dim is [real, imag]
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, 2)
        )

    def _complex_weights(self) -> torch.Tensor:
        return torch.view_as_complex(self.weights.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply 2D spectral convolution.

        Args:
            x: Shape (batch, in_channels, nx, ny).

        Returns:
            Shape (batch, out_channels, nx, ny).
        '''
        batch, _, nx, ny = x.shape

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (batch, in_ch, nx, ny//2+1)

        out_ft = torch.zeros(
            batch, self.out_channels, nx, ny // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        out_ft[:, :, :self.n_modes_x, :self.n_modes_y] = torch.einsum(
            'bixy,ioxy->boxy',
            x_ft[:, :, :self.n_modes_x, :self.n_modes_y],
            self._complex_weights(),
        )

        return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    '''One 2D FNO layer: spectral conv + pointwise bypass + activation.

    Args:
        channels:  Number of channels.
        n_modes_x: Fourier modes in x.
        n_modes_y: Fourier modes in y.
    '''

    def __init__(self, channels: int, n_modes_x: int, n_modes_y: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, n_modes_x, n_modes_y)
        self.bypass        = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.bypass(x))


class FNO2d(Operator):
    '''Fourier Neural Operator for 2D parametric PDEs.

    Maps an initial vorticity field w0 of shape (batch, nx, ny) to the
    solution at time T of the same shape.

    Architecture:
        1. Input projection: (batch, 3, nx, ny) → (batch, channels, nx, ny)
           The 3 input channels are: w0, x-coordinate grid, y-coordinate grid.
        2. FNO blocks: n_layers stacked FNOBlock2d layers.
        3. Output projection: (batch, channels, nx, ny) → (batch, 1, nx, ny).

    The spatial grids are computed dynamically in forward, enabling resolution
    invariance by the same mechanism as FNO1d.

    Args:
        channels:  Hidden channel width.
        n_modes_x: Fourier modes to keep in x direction.
        n_modes_y: Fourier modes to keep in y direction.
        n_layers:  Number of FNO blocks.
    '''

    def __init__(
        self,
        channels:  int = 32,
        n_modes_x: int = 12,
        n_modes_y: int = 12,
        n_layers:  int = 4,
    ) -> None:
        super().__init__()

        # 3 input channels: w0 + x grid + y grid
        self.input_projection  = nn.Conv2d(3, channels, kernel_size=1)
        self.fno_blocks        = nn.Sequential(
            *[FNOBlock2d(channels, n_modes_x, n_modes_y) for _ in range(n_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Map a batch of initial vorticity fields to solutions at time T.

        Args:
            x: Shape (batch, nx, ny).

        Returns:
            Shape (batch, nx, ny).
        '''
        batch, nx, ny = x.shape

        # Build spatial grids dynamically
        grid_x = torch.linspace(0, 1, nx, device=x.device)
        grid_y = torch.linspace(0, 1, ny, device=x.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')

        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)

        x = x.unsqueeze(1)                              # (batch, 1, nx, ny)
        x = torch.cat([x, grid_x, grid_y], dim=1)      # (batch, 3, nx, ny)

        x = self.input_projection(x)                    # (batch, channels, nx, ny)
        x = self.fno_blocks(x)                          # (batch, channels, nx, ny)
        x = self.output_projection(x)                   # (batch, 1, nx, ny)

        return x.squeeze(1)                             # (batch, nx, ny)
