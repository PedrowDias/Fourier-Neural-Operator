'''Plot training and validation loss curves from one or two checkpoints.

Usage:
    python scripts/plot_loss.py --model fno
    python scripts/plot_loss.py --model mlp
    python scripts/plot_loss.py --compare
'''
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str,  default='fno', choices=['mlp', 'fno'])
    parser.add_argument('--compare', action='store_true', help='Plot MLP and FNO side by side')
    parser.add_argument('--epochs',  type=int,  default=100)
    parser.add_argument('--output',  type=Path, default=Path('notebooks/loss_curve.png'))
    return parser.parse_args()


def load_losses(model: str, epochs: int) -> tuple[list, list]:
    path = Path(f'checkpoints/{model}/checkpoint_epoch_{epochs}.pt')
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint['train_losses'], checkpoint['val_losses']


def plot_single(model: str, epochs: int, output: Path) -> None:
    train_losses, val_losses = load_losses(model, epochs)
    epoch_range = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_range, train_losses, label='Train loss (MSE)')
    ax.plot(epoch_range, val_losses,   label='Val loss (relative L2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model.upper()} — Burgers equation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'Saved to {output}')


def plot_compare(epochs: int, output: Path) -> None:
    mlp_train, mlp_val = load_losses('mlp', epochs)
    fno_train, fno_val = load_losses('fno', epochs)
    epoch_range = list(range(1, len(mlp_train) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, train, val, title in [
        (axes[0], mlp_train, mlp_val, 'MLP baseline'),
        (axes[1], fno_train, fno_val, 'Fourier Neural Operator'),
    ]:
        ax.plot(epoch_range, train, label='Train loss (MSE)')
        ax.plot(epoch_range, val,   label='Val loss (relative L2)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Burgers equation — MLP vs FNO', fontsize=13)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'Saved to {output}')


def main() -> None:
    args = parse_args()
    if args.compare:
        plot_compare(args.epochs, args.output)
    else:
        plot_single(args.model, args.epochs, args.output)


if __name__ == '__main__':
    main()
