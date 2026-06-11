import numpy as np
from scipy.integrate import solve_ivp

from fno.solvers.base import Solver


class Burgers1DSolver(Solver):
    '''Finite-difference solver for the viscous 1D Burgers equation:

        u_t + u * u_x = nu * u_xx,    x in [0, 1),  periodic BCs.

    Spatial derivatives are discretised with second-order central differences
    on a uniform periodic grid. Time integration is handed off to scipy's
    `solve_ivp` with the RK45 method.

    The combination of explicit nonlinear advection and implicit-friendly
    diffusion is handled here explicitly; for stiff regimes (very small nu)
    consider switching `method` to 'Radau'.

    Args:
        n_grid: Number of spatial grid points. The grid covers [0, 1) with
                spacing dx = 1 / n_grid.
        nu:     Kinematic viscosity. Controls the diffusion strength and the
                thickness of the shock that forms in the solution.
    '''

    def __init__(self, n_grid: int = 64, nu: float = 0.01) -> None:
        self._n_grid = n_grid
        self._nu = nu
        self._x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
        self._dx = 1.0 / n_grid

    @property
    def grid(self) -> np.ndarray:
        return self._x

    def solve(self, u0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        '''Integrate Burgers equation forward from u0.

        Args:
            u0:     Initial condition, shape (n_grid,).
            t_span: Sorted 1D array of time points including t=0, shape (T,).

        Returns:
            Solution array of shape (T, n_grid).

        Raises:
            ValueError: If u0 has wrong shape or t_span is not sorted.
        '''
        if u0.shape != (self._n_grid,):
            raise ValueError(
                f'u0 must have shape ({self._n_grid},), got {u0.shape}.'
            )
        if not np.all(np.diff(t_span) > 0):
            raise ValueError('t_span must be strictly increasing.')

        result = solve_ivp(
            fun=self._rhs,
            t_span=(t_span[0], t_span[-1]),
            y0=u0,
            method='RK45',
            t_eval=t_span,
            rtol=1e-6,
            atol=1e-8,
        )

        # solve_ivp returns shape (n_grid, T); we want (T, n_grid)
        return result.y.T

    def _rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        '''Right-hand side of the Burgers ODE system.

        Computes  -u * u_x  +  nu * u_xx  using periodic central differences.
        np.roll implements the periodic shift without boundary conditon logic.
        '''
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * self._dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (self._dx ** 2)
        return -u * u_x + self._nu * u_xx
