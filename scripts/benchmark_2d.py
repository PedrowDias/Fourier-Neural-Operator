"""Benchmark FNO2d accuracy and speed vs the Navier-Stokes solver.

Usage:
    python scripts/benchmark_2d.py
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fno.models.fno2d import FNO2d
from fno.solvers.navier_stokes_2d import NavierStokes2DSolver
from fno.training.metrics import relative_l2_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=Path, default=Path("checkpoints/fno2d/checkpoint_epoch_100.pt"))
    parser.add_argument("--data",         type=Path, default=Path("data/ns_64.npz"))
    parser.add_argument("--n-test",       type=int,  default=50)
    parser.add_argument("--sample-sizes", type=int,  nargs="+", default=[1, 5, 10, 20, 50])
    parser.add_argument("--n-trials",     type=int,  default=3)
    parser.add_argument("--output",       type=Path, default=Path("notebooks/benchmark_2d.png"))
    return parser.parse_args()


def load_fno2d(checkpoint_path: Path) -> FNO2d:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FNO2d()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_accuracy(model: FNO2d, data_path: Path, n_test: int) -> float:
    data    = np.load(data_path)
    w0_all  = torch.tensor(data["w0"][-n_test:],  dtype=torch.float32)
    w_T_all = torch.tensor(data["w_T"][-n_test:], dtype=torch.float32)

    # Per-sample normalisation
    w0_norm  = (w0_all  - w0_all.mean(dim=(-2,-1), keepdim=True))  / (w0_all.std(dim=(-2,-1), keepdim=True)  + 1e-8)
    w_T_norm = (w_T_all - w_T_all.mean(dim=(-2,-1), keepdim=True)) / (w_T_all.std(dim=(-2,-1), keepdim=True) + 1e-8)

    total_error = 0.0
    batch_size  = 4
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            pred = model(w0_norm[i:i+batch_size])
            total_error += relative_l2_error(
                pred.flatten(1), w_T_norm[i:i+batch_size].flatten(1)
            ).item()
    return total_error / (n_test // batch_size)


def main():
    args = parse_args()

    model  = load_fno2d(args.checkpoint)
    data   = np.load(args.data)
    n_grid = data["w0"].shape[1]
    nu     = float(data["nu"])
    t_end  = float(data["t_end"])
    solver = NavierStokes2DSolver(n_grid=n_grid, nu=nu)

    # Accuracy
    logger.info("Evaluating accuracy...")
    error = evaluate_accuracy(model, args.data, args.n_test)
    logger.info(f"FNO2d relative L2 error: {error:.4f}")

    # Speed benchmark
    rng = np.random.default_rng(0)
    kx  = np.fft.fftfreq(n_grid, d=1.0/n_grid)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX**2 + KY**2

    def make_ic():
        amp   = np.where(K2==0, 0.0, K2**(-0.75))
        noise = rng.standard_normal((n_grid,n_grid)) + 1j*rng.standard_normal((n_grid,n_grid))
        w = np.real(np.fft.ifft2(amp * noise))
        return w / (w.std() + 1e-8)

    solver_times, fno_times, speedups = [], [], []
    t_span = np.array([0.0, t_end])

    for n in args.sample_sizes:
        ics = [make_ic() for _ in range(n)]

        # Solver timing
        times = []
        for _ in range(args.n_trials):
            start = time.perf_counter()
            for w0 in ics:
                solver.solve(w0, t_span)
            times.append(time.perf_counter() - start)
        ts = float(np.mean(times))

        # FNO timing
        batch = torch.tensor(np.stack(ics), dtype=torch.float32)
        batch = (batch - batch.mean(dim=(-2,-1), keepdim=True)) / (batch.std(dim=(-2,-1), keepdim=True) + 1e-8)
        times = []
        with torch.no_grad():
            for _ in range(args.n_trials):
                start = time.perf_counter()
                _ = model(batch)
                times.append(time.perf_counter() - start)
        tf = float(np.mean(times))

        sp = ts / tf
        solver_times.append(ts)
        fno_times.append(tf)
        speedups.append(sp)
        logger.info(f"N={n:>3} | solver: {ts*1000:7.1f}ms | FNO: {tf*1000:6.2f}ms | speedup: {sp:.1f}x")

    # Print table
    print("\n" + "="*56)
    print(f'{"N samples":<12} {"Solver (ms)":>14} {"FNO (ms)":>12} {"Speedup":>10}')
    print("-"*56)
    for n, ts, tf, sp in zip(args.sample_sizes, solver_times, fno_times, speedups):
        print(f"{n:<12} {ts*1000:>14.1f} {tf*1000:>12.2f} {sp:>9.1f}x")
    print("="*56)
    print(f"\nFNO2d accuracy — relative L2 error: {error:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(args.sample_sizes, [t*1000 for t in solver_times], "o-", color="darkorange", label="NS solver")
    ax1.plot(args.sample_sizes, [t*1000 for t in fno_times],    "o-", color="steelblue",  label="FNO2d")
    ax1.set_xlabel("Number of samples")
    ax1.set_ylabel("Wall-clock time (ms)")
    ax1.set_title("Inference time — 2D Navier-Stokes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(args.sample_sizes, speedups, "o-", color="seagreen")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Number of samples")
    ax2.set_ylabel("Speedup")
    ax2.set_title(f"FNO2d speedup over NS solver\n(val error: {error:.3f} relative L2)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"2D Navier-Stokes benchmark (grid {n_grid}x{n_grid}, nu={nu}, t_end={t_end})", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
