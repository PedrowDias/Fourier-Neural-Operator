import numpy as np
import pytest

from fno.solvers.burgers_1d import Burgers1DSolver
from fno.training.config import DataConfig, TrainingConfig


@pytest.fixture
def default_solver() -> Burgers1DSolver:
    '''A Burgers1DSolver with small grid for fast tests.'''
    return Burgers1DSolver(n_grid=32, nu=0.01)


@pytest.fixture
def default_data_config() -> DataConfig:
    return DataConfig(n_samples=20, n_grid=32, seed=0)


@pytest.fixture
def default_training_config(default_data_config) -> TrainingConfig:
    return TrainingConfig(epochs=2, batch_size=4, data=default_data_config)


@pytest.fixture
def simple_ic(default_solver) -> np.ndarray:
    '''A single sinusoidal initial condition on the solver grid.'''
    x = default_solver.grid
    return np.sin(2 * np.pi * x)
