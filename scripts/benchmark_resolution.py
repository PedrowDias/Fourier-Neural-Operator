'''Benchmark resolution generalization: train at 64, evaluate at 64/128/256.

This script demonstrates the key property of neural operators — resolution
invariance. The FNO is evaluated at resolutions it was never trained on,
while the MLP is shown to fail at any resolution other than its training size.

Usage:
    python scripts/benchmark_resolution.py
    python scripts/benchmark_resolution.py --train-resolution 64 --test-resolutions 64 128 256
'''
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fno.data.dataset import BurgersDataset
from fno.data.generator import generate_dataset
from fno.models.fno1d import FNO1d
from fno.models.mlp import MLP
from fno.solvers.burgers_1d import Burgers1DSolver
from fno.training.metrics import relative_l2_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-resolution',  type=int, default=64)
    parser.add_argument('--test-resolutions',  type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--n-test-samples',    type=int, default=100)
    parser.add_argument('--fno-checkpoint',    type=Path, default=Path('checkpoints/fno/checkpoint_epoch_100.pt'))
    parser.add_argument('--mlp-checkpoint',    type=Path, default=Path('checkpoints/mlp/checkpoint_epoch_100.pt'))
    parser.add_argument('--output',            type=Path, default=Path('notebooks/resolution_benchmark.png'))
    return parser.parse_args()


def load_fno(checkpoint_path: Path, n_grid: int) -> FNO1d:
    '''Load a trained FNO and update its grid buffer for a new resolution.

    The weights are unchanged — only the positional grid buffer is replaced.
    This is valid because the spectral convolution weights are resolution-
    independent; they operate on Fourier coefficients, not on fixed grid indices.
    '''
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = FNO1d(n_grid=64)  # architecture must match training config
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Replace the grid buffer with one matching the new resolution
    model.x_grid = torch.linspace(0, 1, n_grid).reshape(1, 1, n_grid)
    return model


def load_mlp(checkpoint_path: Path) -> MLP:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MLP(n_grid=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_at_resolution(
    model: torch.nn.Module,
    resolution: int,
    n_samples: int,
    tmp_dir: Path,
    seed: int = 99,
) -> float | None:
    '''Generate test data at the given resolution and compute relative L2 error.

    Returns None if the model cannot handle this resolution (e.g. MLP at
    a resolution different from its training size).
    '''
    # Generate test data at this resolution
    data_path = tmp_dir / f'test_{resolution}.npz'
    if not data_path.exists():
        solver = Burgers1DSolver(n_grid=resolution, nu=0.01)
        generate_dataset(solver, n_samples=n_samples, t_end=1.0,
                         output_path=data_path, seed=seed)

    dataset = BurgersDataset(data_path)
    loader = DataLoader(dataset, batch_size=32)

    total_error = 0.0
    n_batches = 0

    with torch.no_grad():
        for u0, u_T in loader:
            try:
                pred = model(u0)
                total_error += relative_l2_error(pred, u_T).item()
                n_batches += 1
            except RuntimeError:
                # MLP input size mismatch — model cannot handle this resolution
                return None

    return total_error / n_batches


def main() -> None:
    args = parse_args()
    tmp_dir = Path('data/benchmark_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = {'FNO': {}, 'MLP': {}}

    for resolution in args.test_resolutions:
        logger.info(f'\nEvaluating at resolution {resolution}...')

        # FNO — load with updated grid for this resolution
        fno = load_fno(args.fno_checkpoint, resolution)
        fno_error = evaluate_at_resolution(fno, resolution, args.n_test_samples, tmp_dir)
        results['FNO'][resolution] = fno_error
        logger.info(f'  FNO  relative L2: {fno_error:.4f}' if fno_error is not None else '  FNO  N/A')

        # MLP — fixed at training resolution
        mlp = load_mlp(args.mlp_checkpoint)
        mlp_error = evaluate_at_resolution(mlp, resolution, args.n_test_samples, tmp_dir)
        results['MLP'][resolution] = mlp_error
        if mlp_error is not None:
            logger.info(f'  MLP  relative L2: {mlp_error:.4f}')
        else:
            logger.info(f'  MLP  relative L2: N/A (input size mismatch)')

    _print_table(results, args.test_resolutions)
    _plot(results, args.test_resolutions, args.output)


def _print_table(results: dict, resolutions: list[int]) -> None:
    print('\n' + '=' * 48)
    print(f'{"Resolution":<14} {"FNO error":>14} {"MLP error":>14}')
    print('-' * 48)
    for res in resolutions:
        fno_val = results['FNO'][res]
        mlp_val = results['MLP'][res]
        fno_str = f'{fno_val:.4f}' if fno_val is not None else 'N/A'
        mlp_str = f'{mlp_val:.4f}' if mlp_val is not None else 'N/A (size mismatch)'
        print(f'{res:<14} {fno_str:>14} {mlp_str:>14}')
    print('=' * 48)


def _plot(results: dict, resolutions: list[int], output: Path) -> None:
    fno_errors = [results['FNO'][r] for r in resolutions]
    mlp_errors = [results['MLP'][r] if results['MLP'][r] is not None else float('nan')
                  for r in resolutions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(resolutions, fno_errors, 'o-', label='FNO (resolution-invariant)')
    ax.plot(resolutions, mlp_errors, 's--', label='MLP (trained at 64 only)',
            color='orange')

    # Mark missing MLP values
    for i, (res, err) in enumerate(zip(resolutions, mlp_errors)):
        if np.isnan(err):
            ax.axvline(x=res, color='orange', linestyle=':', alpha=0.4)
            ax.text(res, ax.get_ylim()[1] * 0.95, 'N/A', ha='center',
                    color='orange', fontsize=9)

    ax.set_xlabel('Evaluation resolution (grid points)')
    ax.set_ylabel('Relative L2 error')
    ax.set_title('Resolution generalization: FNO vs MLP\n(both trained at resolution 64)')
    ax.set_xticks(resolutions)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    logger.info(f'\nPlot saved to {output}')


if __name__ == '__main__':
    main()
