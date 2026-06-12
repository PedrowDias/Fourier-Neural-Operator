# Fourier Neural Operator for the 1D Burgers Equation

A from-scratch PyTorch implementation of the **Fourier Neural Operator (FNO)** applied to the viscous 1D Burgers equation. The FNO learns the solution operator of a parametric PDE — mapping initial conditions to solutions — and generalizes to unseen spatial resolutions without retraining.

Based on [Li et al. (2020), *Fourier Neural Operator for Parametric Partial Differential Equations*](https://arxiv.org/abs/2010.08895).

---

## Results

### Accuracy

| Model | Val relative L2 error |
|---|---|
| MLP baseline | ~0.77 |
| FNO (single resolution) | ~0.29 |
| **FNO (mixed-resolution training)** | **~0.13** |

The FNO achieves **~6× lower error** than the MLP baseline. The MLP overfits — its training loss reaches near zero while validation loss stays flat — because it has no inductive bias about the spatial structure of the problem.

### Resolution invariance

The FNO is trained simultaneously on resolutions 32, 64, and 128. It is then evaluated at resolution 256 **without retraining**.

| Evaluation resolution | FNO error | MLP error |
|---|---|---|
| 32 | 0.59 | N/A (size mismatch) |
| 64 | 0.19 | 1.12 |
| 128 | 0.16 | N/A (size mismatch) |
| 256 | **0.16** | N/A (size mismatch) |

The FNO maintains consistent accuracy at 128 and 256. The MLP cannot run at any resolution other than its training size — a fundamental architectural limitation.

### Inference speed

| N samples | Classical solver | FNO | Speedup |
|---|---|---|---|
| 1 | 9 ms | 5 ms | 1.8× |
| 10 | 109 ms | 2 ms | 64× |
| 100 | 1274 ms | 43 ms | 30× |
| 500 | 6325 ms | 216 ms | **29×** |

The FNO achieves a consistent **~30× speedup** over the classical finite-difference solver at batch sizes of 10 and above.

---

## How it works

### The problem

Solving a PDE classically (e.g. the Burgers equation via finite differences) requires numerical integration for every new initial condition. For 1000 samples this means 1000 independent solver runs — expensive in any design loop or simulation pipeline.

### The idea

Train a neural network on a dataset of (initial condition → solution) pairs produced by the classical solver. After training, the network predicts new solutions in milliseconds.

### Why FNO and not a standard neural network?

A standard MLP or CNN learns to map one fixed-resolution grid to another. Train it on 64-point grids and it only works on 64-point grids. The FNO operates in **frequency space**:

1. Apply FFT to the input
2. Apply a learned linear transform to the lowest Fourier modes
3. Apply inverse FFT back to physical space

Because the learned weights are in frequency space and don't depend on grid size, **the same trained model runs at any resolution**. This is the resolution invariance property.

---

## Project structure

```
src/fno/
├── solvers/        # Classical PDE solvers (finite difference)
├── data/           # Dataset generation, loading, normalisation
├── models/         # MLP baseline and FNO architecture
└── training/       # Trainer, metrics, config

scripts/
├── generate_data.py          # Generate Burgers equation dataset
├── train.py                  # Train MLP or FNO (single or mixed resolution)
├── benchmark_resolution.py   # Resolution generalization benchmark
├── benchmark_speed.py        # Inference speed benchmark
└── plot_loss.py              # Plot training curves

tests/                        # Mirrors src/ structure
```

---

## Quickstart

**Install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Generate data**

```bash
python scripts/generate_data.py                      # 1000 samples at resolution 64
```

**Train**

```bash
python scripts/train.py --model mlp                  # MLP baseline
python scripts/train.py --model fno --mixed-resolution  # FNO on resolutions 32, 64, 128
```

**Benchmark**

```bash
python scripts/benchmark_resolution.py   # Resolution generalization
python scripts/benchmark_speed.py        # Inference speed vs classical solver
```

**Run tests**

```bash
python -m pytest
```

---

## Key implementation details

**Spectral convolution layer** (`src/fno/models/fno1d.py`): The core FNO operation. Applies FFT, multiplies the lowest `n_modes` Fourier coefficients by learned complex weights, then applies inverse FFT. Weights are stored as real tensors of shape `(in_channels, out_channels, n_modes, 2)` and viewed as complex via `torch.view_as_complex`.

**Dynamic spatial grid** (`FNO1d.forward`): The positional grid is computed from the actual input size at each forward pass — not stored as a fixed buffer. This is what enables resolution invariance without any model modification at inference time.

**Unit Gaussian normalisation** (`src/fno/data/transforms.py`): Mixed-resolution training uses per-sample normalisation (zero mean, unit std computed per sample) rather than dataset-level statistics. This is resolution-independent and consistent between training and evaluation.

**Resolution batch sampler** (`scripts/train.py`): Ensures each batch contains samples from one resolution only, since tensors in a batch must share the same shape. Batch order is shuffled across resolutions each epoch.

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- NumPy, SciPy, Matplotlib
