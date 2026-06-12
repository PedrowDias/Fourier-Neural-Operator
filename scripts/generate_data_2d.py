"""Generate a Navier-Stokes 2D dataset and save to disk.

Usage:
    python scripts/generate_data_2d.py
    python scripts/generate_data_2d.py --n-samples 500 --n-grid 64 --output data/ns_64.npz
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from fno.solvers.navier_stokes_2d import NavierStokes2DSolver

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int,   default=200)
    parser.add_argument("--n-grid",    type=int,   default=64)
    parser.add_argument("--nu",        type=float, default=1e-3)
    parser.add_argument("--t-end",     type=float, default=1.0)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--output",    type=Path,  default=Path("data/ns_64.npz"))
    return parser.parse_args()


def sample_grf_2d(n_grid: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a smooth 2D random field via truncated 2D Fourier series.

    Amplitudes decay as 1/(kx^2 + ky^2)^(3/4), giving a smooth field
    with energy concentrated at low wavenumbers.
    """
    kx = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    ky = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2

    amplitude = np.where(K2 == 0, 0.0, K2 ** (-0.75))
    noise = rng.standard_normal((n_grid, n_grid)) + 1j * rng.standard_normal((n_grid, n_grid))
    w_hat = amplitude * noise

    w = np.real(np.fft.ifft2(w_hat))
    return w / (w.std() + 1e-8)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    solver = NavierStokes2DSolver(n_grid=args.n_grid, nu=args.nu)
    t_span = np.array([0.0, args.t_end])

    w0_all  = np.zeros((args.n_samples, args.n_grid, args.n_grid))
    w_T_all = np.zeros((args.n_samples, args.n_grid, args.n_grid))

    for i in range(args.n_samples):
        w0 = sample_grf_2d(args.n_grid, rng)
        solution = solver.solve(w0, t_span)
        w0_all[i]  = solution[0]
        w_T_all[i] = solution[-1]

        if (i + 1) % max(1, args.n_samples // 10) == 0:
            logger.info(f"Generated {i + 1}/{args.n_samples} samples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, w0=w0_all, w_T=w_T_all, nu=args.nu, t_end=args.t_end)
    logger.info(f"Dataset saved to {args.output}.")


if __name__ == "__main__":
    main()
