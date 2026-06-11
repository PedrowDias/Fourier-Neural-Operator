'''Train a model on the Burgers equation dataset.

Usage:
    python scripts/train.py --model mlp
    python scripts/train.py --model fno
    python scripts/train.py --model fno --epochs 200 --lr 1e-3
'''
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fno.data.dataset import BurgersDataset
from fno.models.mlp import MLP
from fno.models.fno1d import FNO1d
from fno.training.config import TrainingConfig
from fno.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=Path,  default=Path('data/burgers.npz'))
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--model',      type=str,   default='fno', choices=['mlp', 'fno'])
    parser.add_argument('--seed',       type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Split indices ---
    dataset_full = BurgersDataset(args.data)
    n = len(dataset_full)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    torch.manual_seed(args.seed)
    indices = torch.randperm(n).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]

    # --- Normalizers fit on training split only ---
    norm_u0, norm_u_T = BurgersDataset.build_normalizers(args.data, train_idx)

    train_ds = BurgersDataset(args.data, norm_u0, norm_u_T)
    val_ds   = BurgersDataset(args.data, norm_u0, norm_u_T)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=args.batch_size)

    # --- Model ---
    n_grid = train_ds[0][0].shape[0]
    if args.model == 'mlp':
        model = MLP(n_grid=n_grid)
    else:
        model = FNO1d(n_grid=n_grid)

    logging.info(f'Model: {model.__class__.__name__} | Parameters: {model.parameter_count():,}')

    # --- Train ---
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        checkpoint_dir=f'checkpoints/{args.model}',
    )
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == '__main__':
    main()
