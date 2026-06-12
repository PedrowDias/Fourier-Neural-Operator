"""Benchmark inference speed: FNO vs classical Burgers solver.

Measures wall-clock time to solve N instances of the Burgers equation,
comparing the classical finite-difference solver against the trained FNO.

Usage:
    python scripts/benchmark_speed.py
    python scripts/benchmark_speed.py --n-trials 5 --sample-sizes 1 10 50 100 500
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from fno.data.generator import _sample_gaussian_random_field
from fno.data.transforms import UnitGaussianNormalizer
from fno.models.fno1d import FNO1d
from fno.solvers.burgers_1d import Burgers1DSolver

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_GRID  = 64
NU      = 0.01
T_END   = 1.0
N_STEPS = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fno-checkpoint", type=Path,
                        default=Path("checkpoints/fno_mixed/checkpoint_epoch_100.pt"))
    parser.add_argument("--sample-sizes", type=int, nargs="+",
                        default=[1, 5, 10, 50, 100, 200, 500])
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("notebooks/speed_benchmark.png"))
    return parser.parse_args()


def load_fno(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FNO1d()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def time_solver(solver, u0_batch, n_trials):
    t_span = np.linspace(0.0, T_END, N_STEPS)
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for u0 in u0_batch:
            solver.solve(u0, t_span)
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


def time_fno(model, u0_batch, n_trials):
    normalizer = UnitGaussianNormalizer()
    u0_tensor = torch.tensor(u0_batch, dtype=torch.float32)
    u0_tensor = normalizer.encode(u0_tensor)
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = model(u0_tensor)
            times.append(time.perf_counter() - start)
    return float(np.mean(times))


def main():
    args = parse_args()
    solver = Burgers1DSolver(n_grid=N_GRID, nu=NU)
    fno = load_fno(args.fno_checkpoint)

    rng = np.random.default_rng(0)
    x = solver.grid
    u0_pool = np.array([_sample_gaussian_random_field(x, rng)
                        for _ in range(max(args.sample_sizes))])

    solver_times, fno_times, speedups = [], [], []

    for n in args.sample_sizes:
        u0_batch = u0_pool[:n]
        ts = time_solver(solver, u0_batch, args.n_trials)
        tf = time_fno(fno, u0_batch, args.n_trials)
        sp = ts / tf
        solver_times.append(ts)
        fno_times.append(tf)
        speedups.append(sp)
        logger.info(
            f"N={n:>4} | solver: {ts*1000:7.1f}ms | "
            f"FNO: {tf*1000:7.2f}ms | speedup: {sp:.1f}x"
        )

    _print_table(args.sample_sizes, solver_times, fno_times, speedups)
    _plot(args.sample_sizes, solver_times, fno_times, speedups, args.output)


def _print_table(sizes, solver_times, fno_times, speedups):
    print("\n" + "=" * 60)
    print(f'{"N samples":<12} {"Solver (ms)":>14} {"FNO (ms)":>12} {"Speedup":>10}')
    print("-" * 60)
    for n, ts, tf, sp in zip(sizes, solver_times, fno_times, speedups):
        print(f"{n:<12} {ts*1000:>14.1f} {tf*1000:>12.2f} {sp:>9.1f}x")
    print("=" * 60)


def _plot(sizes, solver_times, fno_times, speedups, output):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sizes, [t * 1000 for t in solver_times], "o-", color="darkorange", label="Classical solver")
    ax1.plot(sizes, [t * 1000 for t in fno_times],    "o-", color="steelblue",  label="FNO")
    ax1.set_xlabel("Number of samples")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Inference time vs number of samples")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sizes, speedups, "o-", color="seagreen")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Number of samples")
    ax2.set_ylabel("Speedup (solver time / FNO time)")
    ax2.set_title("FNO speedup over classical solver")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Speed benchmark: FNO vs classical finite-difference solver\n"
        f"(Burgers equation, grid size {N_GRID}, t_end={T_END})", fontsize=12
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    logger.info(f"\nPlot saved to {output}")


if __name__ == "__main__":
    main()
