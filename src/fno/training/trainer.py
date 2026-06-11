import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fno.models.base import Operator
from fno.training.config import TrainingConfig
from fno.training.metrics import relative_l2_error

logger = logging.getLogger(__name__)


class Trainer:
    '''Owns the training loop for any Operator model.

    Deliberately knows nothing about which model it is training — it depends
    only on the Operator interface. Swapping MLP for FNO requires zero changes
    here.

    Responsibilities:
        - Train loop with gradient updates
        - Validation loop with metric logging
        - Periodic checkpointing
        - Reproducibility via seeding

    Args:
        model:      Any Operator instance.
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        config:     TrainingConfig dataclass.
    '''

    def __init__(
        self,
        model: Operator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self) -> None:
        '''Run the full training loop for config.epochs epochs.'''
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(
                f'Epoch {epoch}/{self.config.epochs} — '
                f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}'
            )

            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for u0, u_T in self.train_loader:
            u0 = u0.to(self.device)
            u_T = u_T.to(self.device)

            self.optimizer.zero_grad()
            prediction = self.model(u0)
            loss = self.loss_fn(prediction, u_T)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        self.model.eval()
        total_error = 0.0

        with torch.no_grad():
            for u0, u_T in self.val_loader:
                u0 = u0.to(self.device)
                u_T = u_T.to(self.device)
                prediction = self.model(u0)
                total_error += relative_l2_error(prediction, u_T).item()

        return total_error / len(self.val_loader)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        logger.info(f'Checkpoint saved to {path}.')
