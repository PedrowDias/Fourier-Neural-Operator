import numpy as np

from fno.solvers.base import Solver


class NavierStokes2DSolver(Solver):
    '''Pseudo-spectral solver for 2D incompressible Navier-Stokes in vorticity form.

    Solves:
        dw/dt + u * dw/dx + v * dw/dy = nu * (d²w/dx² + d²w/dy²) + f

    where w is vorticity, (u, v) is velocity recovered from the stream function
    via the Poisson equation, nu is kinematic viscosity, and f is an optional
    forcing term.

    The pseudo-spectral method computes spatial derivatives in frequency space
    (exact to machine precision) and the nonlinear advection term in physical
    space. A 2/3 dealiasing filter removes aliasing errors from the nonlinear
    product.

    Time integration uses the 4th-order Runge-Kutta scheme. The internal
    timestep dt is chosen automatically to satisfy the CFL condition, and
    multiple substeps are taken between each pair of t_span points.

    Args:
        n_grid:   Number of grid points in each spatial direction.
        nu:       Kinematic viscosity.
        forcing:  Optional 2D forcing array of shape (n_grid, n_grid).
        cfl:      CFL safety factor for automatic timestep selection.
    '''

    def __init__(
        self,
        n_grid:  int = 64,
        nu:      float = 1e-3,
        forcing: np.ndarray | None = None,
        cfl:     float = 0.5,
    ) -> None:
        self._n_grid  = n_grid
        self._nu      = nu
        self._cfl     = cfl
        self._forcing = forcing if forcing is not None else np.zeros((n_grid, n_grid))

        # Spatial grid on [0, 2pi) x [0, 2pi)
        x = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        self._x, self._y = np.meshgrid(x, x, indexing='ij')
        self._dx = 2 * np.pi / n_grid

        # Wavenumber arrays
        k = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
        self._kx, self._ky = np.meshgrid(k, k, indexing='ij')
        self._k2     = self._kx ** 2 + self._ky ** 2
        self._k2_inv = np.where(self._k2 == 0, 0.0, 1.0 / np.where(self._k2 == 0, 1.0, self._k2))

        # 2/3 dealiasing filter
        k_max = n_grid // 3
        self._dealias = (np.abs(self._kx) < k_max) & (np.abs(self._ky) < k_max)

        # Diffusion operator in frequency space
        self._diff = -self._nu * self._k2

    @property
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        return self._x, self._y

    def solve(self, u0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        '''Integrate the vorticity field forward in time.

        Args:
            u0:     Initial vorticity, shape (n_grid, n_grid).
            t_span: Sorted 1D array of time points, shape (T,).

        Returns:
            Solution array of shape (T, n_grid, n_grid).
        '''
        if u0.shape != (self._n_grid, self._n_grid):
            raise ValueError(
                f'u0 must have shape ({self._n_grid}, {self._n_grid}), got {u0.shape}.'
            )
        if not np.all(np.diff(t_span) > 0):
            raise ValueError('t_span must be strictly increasing.')

        w = u0.copy().astype(np.float64)
        solution = np.zeros((len(t_span), self._n_grid, self._n_grid))
        solution[0] = w

        t = t_span[0]
        for i in range(1, len(t_span)):
            w = self._integrate(w, t, t_span[i])
            solution[i] = w
            t = t_span[i]

        return solution

    def _integrate(self, w: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        '''Integrate from t_start to t_end using adaptive substeps.'''
        t = t_start
        while t < t_end:
            dt = self._compute_dt(w, t_end - t)
            w  = self._rk4_step(w, dt)
            t += dt
        return w

    def _compute_dt(self, w: np.ndarray, dt_max: float) -> float:
        '''Compute a stable timestep from the CFL condition.

        Uses the maximum velocity magnitude and the diffusion stability limit.
        '''
        w_hat   = np.fft.fft2(w)
        psi_hat = w_hat * self._k2_inv
        u = np.real(np.fft.ifft2( 1j * self._ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self._kx * psi_hat))

        u_max = max(np.abs(u).max(), np.abs(v).max(), 1e-8)
        dt_cfl  = self._cfl * self._dx / u_max
        dt_diff = self._cfl * self._dx ** 2 / (self._nu + 1e-10)
        return float(min(dt_cfl, dt_diff, dt_max))

    def _rhs(self, w: np.ndarray) -> np.ndarray:
        w_hat = np.fft.fft2(w)

        psi_hat  = w_hat * self._k2_inv
        u_hat    =  1j * self._ky * psi_hat
        v_hat    = -1j * self._kx * psi_hat
        dwdx_hat =  1j * self._kx * w_hat
        dwdy_hat =  1j * self._ky * w_hat

        u    = np.real(np.fft.ifft2(u_hat))
        v    = np.real(np.fft.ifft2(v_hat))
        dwdx = np.real(np.fft.ifft2(dwdx_hat))
        dwdy = np.real(np.fft.ifft2(dwdy_hat))

        advection_hat = np.fft.fft2(u * dwdx + v * dwdy)
        forcing_hat   = np.fft.fft2(self._forcing)
        rhs_hat = (self._diff * w_hat - advection_hat + forcing_hat) * self._dealias

        return np.real(np.fft.ifft2(rhs_hat))

    def _rk4_step(self, w: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._rhs(w)
        k2 = self._rhs(w + 0.5 * dt * k1)
        k3 = self._rhs(w + 0.5 * dt * k2)
        k4 = self._rhs(w + dt * k3)
        return w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
