from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Operator(ABC, nn.Module):
    '''Abstract interface for all neural operator models.

    A neural operator learns a mapping between function spaces:
    given a discretised input function (e.g. initial condition),
    it predicts a discretised output function (e.g. solution at time T).

    Concretely this is a torch.nn.Module whose forward pass takes a
    batch of input functions and returns a batch of output functions
    of the same spatial resolution.

    Subclasses must implement `forward`. Everything else — training,
    checkpointing, benchmarking — depends only on this interface.
    '''

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Map a batch of input functions to output functions.

        Args:
            x: Input tensor of shape (batch, n_grid) for 1D operators,
               or (batch, nx, ny) for 2D operators.

        Returns:
            Output tensor of the same spatial shape as x.
        '''

    def parameter_count(self) -> int:
        '''Total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
