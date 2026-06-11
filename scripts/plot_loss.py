'''Plot training and validation loss curves from a checkpoint.

Usage:
    python scripts/plot_loss.py
    python scripts/plot_loss.py --checkpoint checkpoints/checkpoint_epoch_100.pt --output notebooks/loss_curve.png
'''
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/checkpoint_epoch_100.pt'))
    parser.add_argument('--output',     type=Path, default=Path('notebooks/loss_curve.png'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    train_losses = checkpoint['train_losses']
    val_losses   = checkpoint['val_losses']
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label='Train loss (MSE)')
    ax.plot(epochs, val_losses,   label='Val loss (relative L2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MLP baseline — Burgers equation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
