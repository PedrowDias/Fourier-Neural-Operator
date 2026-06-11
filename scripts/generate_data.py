'''Generate the Burgers equation dataset and save it to disk.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n-samples 2000 --n-grid 128 --output data/burgers_128.npz
'''
import argparse
import logging
from pathlib import Path

from fno.data.generator import generate_dataset
from fno.solvers.burgers_1d import Burgers1DSolver

logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Burgers equation dataset.')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--n-grid',    type=int, default=64)
    parser.add_argument('--nu',        type=float, default=0.01)
    parser.add_argument('--t-end',     type=float, default=1.0)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--output',    type=Path, default=Path('data/burgers.npz'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solver = Burgers1DSolver(n_grid=args.n_grid, nu=args.nu)
    generate_dataset(
        solver=solver,
        n_samples=args.n_samples,
        t_end=args.t_end,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
