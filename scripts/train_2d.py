"""Train FNO2d on the Navier-Stokes dataset.

Usage:
    python scripts/train_2d.py
    python scripts/train_2d.py --data data/ns_64.npz --epochs 100
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fno.models.fno2d import FNO2d
from fno.training.config import TrainingConfig
from fno.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class NavierStokesDataset(Dataset):
    """Dataset for 2D Navier-Stokes (w0, w_T) pairs with per-sample normalisation."""

    def __init__(self, w0: np.ndarray, w_T: np.ndarray) -> None:
        self.w0  = torch.tensor(w0,  dtype=torch.float32)
        self.w_T = torch.tensor(w_T, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.w0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w0  = self.w0[idx]
        w_T = self.w_T[idx]
        # Per-sample normalisation — resolution-independent
        w0  = (w0  - w0.mean())  / (w0.std()  + 1e-8)
        w_T = (w_T - w_T.mean()) / (w_T.std() + 1e-8)
        return w0, w_T


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=Path,  default=Path("data/ns_64.npz"))
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    data    = np.load(args.data)
    w0_all  = data["w0"]
    w_T_all = data["w_T"]
    n       = len(w0_all)

    torch.manual_seed(args.seed)
    idx     = torch.randperm(n).numpy()
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_ds = NavierStokesDataset(w0_all[idx[:n_train]],          w_T_all[idx[:n_train]])
    val_ds   = NavierStokesDataset(w0_all[idx[n_train:n_train+n_val]], w_T_all[idx[n_train:n_train+n_val]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = FNO2d()
    logger.info(f"Model: FNO2d | Parameters: {model.parameter_count():,}")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        checkpoint_dir="checkpoints/fno2d",
    )
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
