'''Train a model on the Burgers equation dataset.

Usage:
    python scripts/train.py --model mlp
    python scripts/train.py --model fno
    python scripts/train.py --model fno --mixed-resolution
'''
import argparse
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, BatchSampler, SequentialSampler

from fno.data.dataset import BurgersDataset, build_mixed_resolution_dataset
from fno.data.generator import generate_dataset
from fno.models.mlp import MLP
from fno.models.fno1d import FNO1d
from fno.solvers.burgers_1d import Burgers1DSolver
from fno.training.config import TrainingConfig
from fno.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MIXED_RESOLUTIONS = [32, 64, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',              type=Path,  default=Path('data/burgers.npz'))
    parser.add_argument('--epochs',            type=int,   default=100)
    parser.add_argument('--batch-size',        type=int,   default=32)
    parser.add_argument('--lr',                type=float, default=1e-3)
    parser.add_argument('--model',             type=str,   default='fno', choices=['mlp', 'fno'])
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--mixed-resolution',  action='store_true',
                        help='Train FNO on data at resolutions 32, 64, and 128 simultaneously.')
    return parser.parse_args()


class ResolutionBatchSampler:
    '''Yields batches where every sample has the same resolution.

    Mixed-resolution training requires that all samples in a batch share the
    same grid size, since tensors in a batch must be the same shape. This
    sampler groups indices by resolution and yields homogeneous batches,
    shuffling the order of batches across resolutions each epoch.

    Args:
        dataset_sizes: List of dataset sizes, one per resolution, in the
                       same order as the ConcatDataset.
        batch_size:    Number of samples per batch.
        shuffle:       Whether to shuffle batch order each epoch.
    '''

    def __init__(
        self,
        dataset_sizes: list[int],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Build index ranges for each resolution's slice in the ConcatDataset
        self.batches: list[list[int]] = []
        offset = 0
        for size in dataset_sizes:
            indices = list(range(offset, offset + size))
            if shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])
            offset += size

        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def ensure_mixed_resolution_data(resolutions: list[int], n_samples: int, seed: int) -> list[Path]:
    '''Generate datasets at each resolution if they don't already exist.'''
    paths = []
    for res in resolutions:
        path = Path(f'data/burgers_{res}.npz')
        if not path.exists():
            logger.info(f'Generating dataset at resolution {res}...')
            solver = Burgers1DSolver(n_grid=res, nu=0.01)
            generate_dataset(solver, n_samples=n_samples, t_end=1.0,
                             output_path=path, seed=seed)
        paths.append(path)
    return paths


def build_single_resolution_loaders(
    data_path: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    dataset_full = BurgersDataset(data_path)
    n = len(dataset_full)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    torch.manual_seed(seed)
    indices   = torch.randperm(n).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]

    norm_u0, norm_u_T = BurgersDataset.build_normalizers(data_path, train_idx)

    train_ds = BurgersDataset(data_path, norm_u0, norm_u_T)
    val_ds   = BurgersDataset(data_path, norm_u0, norm_u_T)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=batch_size)
    return train_loader, val_loader


def build_mixed_resolution_loaders(
    resolutions: list[int],
    n_samples: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    paths = ensure_mixed_resolution_data(resolutions, n_samples, seed)
    train_dataset = build_mixed_resolution_dataset(paths)
    val_dataset   = build_mixed_resolution_dataset(paths)

    dataset_sizes = [len(BurgersDataset(p)) for p in paths]
    train_sampler = ResolutionBatchSampler(dataset_sizes, batch_size, shuffle=True)
    val_sampler   = ResolutionBatchSampler(dataset_sizes, batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler)
    return train_loader, val_loader


def main() -> None:
    args = parse_args()

    if args.mixed_resolution and args.model != 'fno':
        raise ValueError('Mixed-resolution training is only supported for the FNO.')

    if args.mixed_resolution:
        logger.info(f'Mixed-resolution training at resolutions: {MIXED_RESOLUTIONS}')
        train_loader, val_loader = build_mixed_resolution_loaders(
            MIXED_RESOLUTIONS, n_samples=1000, batch_size=args.batch_size, seed=args.seed
        )
        n_grid = 64  # reference resolution for logging only
    else:
        train_loader, val_loader = build_single_resolution_loaders(
            args.data, args.batch_size, args.seed
        )
        n_grid = next(iter(train_loader))[0].shape[1]

    # Model
    if args.model == 'mlp':
        model = MLP(n_grid=n_grid)
    else:
        model = FNO1d()  # no n_grid needed — computed dynamically

    logger.info(f'Model: {model.__class__.__name__} | Parameters: {model.parameter_count():,}')

    suffix = 'mixed' if args.mixed_resolution else 'single'
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        checkpoint_dir=f'checkpoints/{args.model}_{suffix}',
    )
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == '__main__':
    main()
