'''Benchmark resolution generalization: train on 32/64/128, evaluate on 32/64/128/256.

Usage:
    python scripts/benchmark_resolution.py
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
from fno.data.transforms import UnitGaussianNormalizer
from fno.models.fno1d import FNO1d
from fno.models.mlp import MLP
from fno.solvers.burgers_1d import Burgers1DSolver
from fno.training.metrics import relative_l2_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-resolutions', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--n-test-samples',   type=int, default=100)
    parser.add_argument('--fno-checkpoint',   type=Path, default=Path('checkpoints/fno_mixed/checkpoint_epoch_100.pt'))
    parser.add_argument('--mlp-checkpoint',   type=Path, default=Path('checkpoints/mlp_single/checkpoint_epoch_100.pt'))
    parser.add_argument('--output',           type=Path, default=Path('notebooks/resolution_benchmark.png'))
    return parser.parse_args()


def load_fno(checkpoint_path: Path) -> FNO1d:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = FNO1d()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_mlp(checkpoint_path: Path) -> MLP:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MLP(n_grid=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_test_data(resolution: int, n_samples: int, tmp_dir: Path, seed: int = 99) -> Path:
    data_path = tmp_dir / f'test_{resolution}.npz'
    if not data_path.exists():
        solver = Burgers1DSolver(n_grid=resolution, nu=0.01)
        generate_dataset(solver, n_samples=n_samples, t_end=1.0,
                         output_path=data_path, seed=seed)
    return data_path


def evaluate_fno(model: FNO1d, resolution: int, n_samples: int, tmp_dir: Path) -> float:
    '''Evaluate FNO with unit normalisation — same as training.'''
    data_path = get_test_data(resolution, n_samples, tmp_dir)
    norm = UnitGaussianNormalizer()
    dataset = BurgersDataset(data_path, normalizer_u0=norm, normalizer_u_T=norm)
    loader  = DataLoader(dataset, batch_size=32)

    total_error = 0.0
    n_batches   = 0
    with torch.no_grad():
        for u0, u_T in loader:
            pred = model(u0)
            total_error += relative_l2_error(pred, u_T).item()
            n_batches += 1
    return total_error / n_batches


def evaluate_mlp(model: MLP, resolution: int, n_samples: int, tmp_dir: Path) -> float | None:
    '''Evaluate MLP — returns None if resolution doesn't match training size.'''
    data_path = get_test_data(resolution, n_samples, tmp_dir)
    dataset = BurgersDataset(data_path)
    loader  = DataLoader(dataset, batch_size=32)

    total_error = 0.0
    n_batches   = 0
    with torch.no_grad():
        for u0, u_T in loader:
            try:
                pred = model(u0)
                total_error += relative_l2_error(pred, u_T).item()
                n_batches += 1
            except RuntimeError:
                return None
    return total_error / n_batches


def main() -> None:
    args = parse_args()
    tmp_dir = Path('data/benchmark_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fno = load_fno(args.fno_checkpoint)
    mlp = load_mlp(args.mlp_checkpoint)

    fno_errors: dict[int, float]        = {}
    mlp_errors: dict[int, float | None] = {}

    for resolution in args.test_resolutions:
        logger.info(f'\nEvaluating at resolution {resolution}...')

        fno_errors[resolution] = evaluate_fno(fno, resolution, args.n_test_samples, tmp_dir)
        mlp_errors[resolution] = evaluate_mlp(mlp, resolution, args.n_test_samples, tmp_dir)

        logger.info(f'  FNO: {fno_errors[resolution]:.4f}')
        mlp_val = mlp_errors[resolution]
        logger.info(f'  MLP: {mlp_val:.4f}' if mlp_val is not None else '  MLP: N/A (size mismatch)')

    _print_table(fno_errors, mlp_errors, args.test_resolutions)
    _plot(fno_errors, mlp_errors, args.test_resolutions, args.output)


def _print_table(fno_errors, mlp_errors, resolutions):
    print('\n' + '=' * 56)
    print(f'{"Resolution":<14} {"FNO (mixed-res)":>20} {"MLP (64 only)":>16}')
    print('-' * 56)
    for res in resolutions:
        fno_str = f'{fno_errors[res]:.4f}'
        mlp_val = mlp_errors[res]
        mlp_str = f'{mlp_val:.4f}' if mlp_val is not None else 'N/A'
        print(f'{res:<14} {fno_str:>20} {mlp_str:>16}')
    print('=' * 56)


def _plot(fno_errors, mlp_errors, resolutions, output):
    fno_y = [fno_errors[r] for r in resolutions]
    mlp_y = [mlp_errors[r] if mlp_errors[r] is not None else float('nan') for r in resolutions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(resolutions, fno_y, 'o-', color='steelblue', label='FNO (trained on 32, 64, 128)')
    ax.plot(resolutions, mlp_y, 's--', color='darkorange', label='MLP (trained on 64 only)')

    for res, val in zip(resolutions, mlp_y):
        if np.isnan(val):
            ax.text(res, max(fno_y) * 0.95, 'N/A', ha='center', color='darkorange', fontsize=9)

    ax.set_xlabel('Evaluation resolution (grid points)')
    ax.set_ylabel('Relative L2 error')
    ax.set_title('Resolution generalization benchmark\nFNO trained on resolutions 32, 64, 128 simultaneously')
    ax.set_xticks(resolutions)
    ax.legend()
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    logger.info(f'\nPlot saved to {output}')


if __name__ == '__main__':
    main()
