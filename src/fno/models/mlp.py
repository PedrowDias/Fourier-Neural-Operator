import torch
import torch.nn as nn

from fno.models.base import Operator


class MLP(Operator):
    '''Multilayer perceptron baseline for learning the Burgers solution operator.

    Maps a discretised initial condition u0 of shape (batch, n_grid) to the
    solution u_T of the same shape. Every spatial point is treated as an
    independent feature — the model has no notion of spatial structure or
    ordering. This is intentional: it sets a baseline that the FNO, which
    explicitly uses spatial structure via Fourier transforms, should beat.

    Architecture:
        Linear(n_grid, hidden_dim)
        [ReLU → Linear(hidden_dim, hidden_dim)] × (n_layers - 1)
        Linear(hidden_dim, n_grid)

    Args:
        n_grid:     Number of spatial grid points (input and output size).
        hidden_dim: Width of each hidden layer.
        n_layers:   Total number of linear layers, including input and output.
                    Must be >= 2.
    '''

    def __init__(
        self,
        n_grid: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()

        if n_layers < 2:
            raise ValueError(f'n_layers must be >= 2, got {n_layers}.')

        layers: list[nn.Module] = [nn.Linear(n_grid, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, n_grid))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Map a batch of initial conditions to predicted solutions.

        Args:
            x: Shape (batch, n_grid).

        Returns:
            Shape (batch, n_grid).
        '''
        return self.net(x)
