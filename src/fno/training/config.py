from dataclasses import dataclass, field


@dataclass
class DataConfig:
    '''Parameters controlling dataset generation.'''
    n_samples: int = 1000
    n_grid: int = 64
    nu: float = 0.01
    t_end: float = 1.0
    n_time_steps: int = 100
    seed: int = 42


@dataclass
class TrainingConfig:
    '''Hyperparameters and infrastructure settings for a training run.'''
    # Optimisation
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Data split
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    # test_fraction is implicitly 1 - train - val

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 10

    # Reproducibility
    seed: int = 42

    # Device — 'cpu', 'cuda', or 'mps'
    device: str = 'cpu'

    data: DataConfig = field(default_factory=DataConfig)
