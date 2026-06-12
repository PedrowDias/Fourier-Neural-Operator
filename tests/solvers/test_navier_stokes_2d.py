import numpy as np
import pytest

from fno.solvers.navier_stokes_2d import NavierStokes2DSolver


class TestNavierStokes2DSolver:

    def test_output_shape(self):
        solver = NavierStokes2DSolver(n_grid=32, nu=1e-3)
        w0 = np.random.randn(32, 32)
        t_span = np.linspace(0, 0.1, 5)
        assert solver.solve(w0, t_span).shape == (5, 32, 32)

    def test_initial_condition_preserved(self):
        solver = NavierStokes2DSolver(n_grid=32, nu=1e-3)
        w0 = np.random.randn(32, 32)
        t_span = np.linspace(0, 0.1, 5)
        solution = solver.solve(w0, t_span)
        np.testing.assert_allclose(solution[0], w0, atol=1e-10)

    def test_wrong_shape_raises(self):
        solver = NavierStokes2DSolver(n_grid=32)
        with pytest.raises(ValueError, match='u0 must have shape'):
            solver.solve(np.zeros((16, 32)), np.linspace(0, 0.1, 3))

    def test_unsorted_t_span_raises(self):
        solver = NavierStokes2DSolver(n_grid=32)
        with pytest.raises(ValueError, match='strictly increasing'):
            solver.solve(np.zeros((32, 32)), np.array([0.0, 0.5, 0.3]))

    def test_diffusion_single_mode(self):
        '''A single Fourier mode w = sin(kx) decays as exp(-nu * k^2 * t).

        With no nonlinear advection (the mode self-advects trivially at zero
        velocity since u = dψ/dy and ψ depends only on x), the solution is
        purely diffusive and the amplitude should match the analytical result.
        '''
        nu = 0.1
        solver = NavierStokes2DSolver(n_grid=32, nu=nu)
        x, _ = solver.grid
        k = 1  # wavenumber
        w0 = np.sin(k * x)  # depends only on x → zero y-velocity → no advection

        t_end = 0.5
        t_span = np.array([0.0, t_end])
        solution = solver.solve(w0, t_span)

        # Analytical decay: amplitude multiplied by exp(-nu * k^2 * t)
        expected_amplitude = np.exp(-nu * k ** 2 * t_end)
        actual_amplitude   = np.max(np.abs(solution[-1])) / np.max(np.abs(w0))
        np.testing.assert_allclose(actual_amplitude, expected_amplitude, rtol=0.05)

    def test_zero_viscosity_conserves_enstrophy_approx(self):
        '''With zero viscosity and no forcing, enstrophy should be
        approximately conserved over short times.'''
        solver = NavierStokes2DSolver(n_grid=32, nu=0.0)
        np.random.seed(1)
        w0 = np.random.randn(32, 32)
        t_span = np.linspace(0, 0.05, 5)
        solution = solver.solve(w0, t_span)
        enstrophy_0 = np.mean(solution[0] ** 2)
        enstrophy_T = np.mean(solution[-1] ** 2)
        np.testing.assert_allclose(enstrophy_0, enstrophy_T, rtol=0.05)
