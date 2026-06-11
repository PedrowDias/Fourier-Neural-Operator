import numpy as np
import pytest

from fno.solvers.burgers_1d import Burgers1DSolver


class TestBurgers1DSolver:

    def test_output_shape(self, default_solver, simple_ic):
        t_span = np.linspace(0.0, 0.5, 11)
        solution = default_solver.solve(simple_ic, t_span)
        assert solution.shape == (11, 32)

    def test_initial_condition_preserved(self, default_solver, simple_ic):
        t_span = np.linspace(0.0, 0.5, 11)
        solution = default_solver.solve(simple_ic, t_span)
        np.testing.assert_allclose(solution[0], simple_ic, atol=1e-6)

    def test_conservation_with_periodic_bcs(self, default_solver, simple_ic):
        '''Mean of u should be conserved under periodic BCs (no sources/sinks).'''
        t_span = np.linspace(0.0, 1.0, 21)
        solution = default_solver.solve(simple_ic, t_span)
        initial_mean = solution[0].mean()
        final_mean = solution[-1].mean()
        np.testing.assert_allclose(initial_mean, final_mean, atol=1e-4)

    def test_wrong_u0_shape_raises(self, default_solver):
        bad_u0 = np.zeros(10)  # solver expects shape (32,)
        t_span = np.linspace(0.0, 0.1, 5)
        with pytest.raises(ValueError, match='u0 must have shape'):
            default_solver.solve(bad_u0, t_span)

    def test_unsorted_t_span_raises(self, default_solver, simple_ic):
        bad_t = np.array([0.0, 0.5, 0.3, 1.0])
        with pytest.raises(ValueError, match='strictly increasing'):
            default_solver.solve(simple_ic, bad_t)

    def test_diffusion_decreases_amplitude(self, default_solver):
        '''High-viscosity solver should damp the initial amplitude significantly.'''
        high_nu_solver = Burgers1DSolver(n_grid=32, nu=0.1)
        x = high_nu_solver.grid
        u0 = np.sin(2 * np.pi * x)
        t_span = np.linspace(0.0, 1.0, 21)
        solution = high_nu_solver.solve(u0, t_span)
        assert solution[-1].max() < u0.max()
