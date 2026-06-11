import torch
from torch.utils.data import DataLoader, TensorDataset

from fno.models.mlp import MLP
from fno.training.config import TrainingConfig
from fno.training.trainer import Trainer


def _make_loaders(n_grid: int = 32, n_samples: int = 16, batch_size: int = 4):
    u0 = torch.randn(n_samples, n_grid)
    u_T = torch.randn(n_samples, n_grid)
    ds = TensorDataset(u0, u_T)
    train_loader = DataLoader(ds, batch_size=batch_size)
    val_loader   = DataLoader(ds, batch_size=batch_size)
    return train_loader, val_loader


class TestTrainer:

    def test_train_loss_recorded(self):
        model = MLP(n_grid=32)
        train_loader, val_loader = _make_loaders()
        config = TrainingConfig(epochs=3, batch_size=4)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        assert len(trainer.train_losses) == 3

    def test_val_loss_recorded(self):
        model = MLP(n_grid=32)
        train_loader, val_loader = _make_loaders()
        config = TrainingConfig(epochs=3, batch_size=4)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        assert len(trainer.val_losses) == 3

    def test_loss_decreases_on_easy_task(self):
        '''Model should overfit a tiny dataset — loss should drop.'''
        torch.manual_seed(0)
        model = MLP(n_grid=32, hidden_dim=128)
        train_loader, val_loader = _make_loaders(n_samples=8, batch_size=8)
        config = TrainingConfig(epochs=50, batch_size=8, learning_rate=1e-2)
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        assert trainer.train_losses[-1] < trainer.train_losses[0]
