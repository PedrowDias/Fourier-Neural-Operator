from abc import ABC, abstractmethod

import numpy as np


class Solver(ABC):
    '''Abstract interface for all classical PDE solvers.

    Every concrete solver must implement `solve`, which maps an initial
    condition u0 defined on a spatial grid to a solution trajectory over
    a sequence of time points.

    This interface lets the rest of the codebase (data generation, benchmarks)
    depend on the abstraction rather than any specific solver — satisfying the
    Open/Closed principle: adding a new solver never touches existing code.
    '''

    @abstractmethod
    def solve(self, u0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        '''Solve the PDE forward in time from an initial condition.

        Args:
            u0:     Initial condition, shape (N,) for 1D or (Nx, Ny) for 2D,
                    defined on the spatial grid returned by `grid`.
            t_span: 1D array of time points at which to record the solution,
                    including t=0. Shape (T,).

        Returns:
            Solution array of shape (T, *spatial_shape), where
            solution[0] == u0.
        '''

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        '''Spatial grid coordinates on which the solver operates.

        Returns:
            1D array of shape (N,) for 1D solvers,
            or a tuple (x, y) of 2D coordinate arrays for 2D solvers.
        '''
