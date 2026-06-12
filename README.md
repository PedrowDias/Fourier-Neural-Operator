# Fourier Neural Operator

A from-scratch PyTorch implementation of the **Fourier Neural Operator (FNO)** applied to two benchmark PDEs:

- **1D Burgers equation** — viscous shock formation, resolution-invariance benchmark
- **2D Navier-Stokes equation** — incompressible turbulent flow, pseudo-spectral solver

The FNO learns the solution operator of a parametric PDE — mapping initial conditions to solutions — and generalizes to unseen spatial resolutions without retraining.

Based on [Li et al. (2020), *Fourier Neural Operator for Parametric Partial Differential Equations*](https://arxiv.org/abs/2010.08895).

---

## Results

### 1D Burgers equation

**Accuracy**

| Model | Val relative L2 error |
|---|---|
| MLP baseline | ~0.77 |
| FNO (single resolution) | ~0.29 |
| **FNO (mixed-resolution training)** | **~0.13** |

The FNO achieves **~6× lower error** than the MLP baseline. The MLP overfits — training loss reaches near zero while validation loss stays flat — because it has no inductive bias about the spatial structure of the problem.

**Resolution invariance**

The FNO is trained simultaneously on resolutions 32, 64, and 128, then evaluated at resolution 256 without retraining.

| Evaluation resolution | FNO error | MLP error |
|---|---|---|
| 32 | 0.59 | N/A (size mismatch) |
| 64 | 0.19 | 1.12 |
| 128 | 0.16 | N/A (size mismatch) |
| **256** | **0.16** | N/A (size mismatch) |

The FNO maintains consistent accuracy at 128 and 256. The MLP cannot run at any resolution other than its training size.

**Inference speed**

| N samples | Classical solver | FNO | Speedup |
|---|---|---|---|
| 1 | 9 ms | 5 ms | 1.8× |
| 10 | 109 ms | 2 ms | 64× |
| 100 | 1274 ms | 43 ms | 30× |
| 500 | 6325 ms | 216 ms | **~30×** |

---

### 2D Navier-Stokes equation

**Accuracy**

| Training samples | Val relative L2 error |
|---|---|
| 200 | 0.36 |
| **1000** | **0.34** |

**Inference speed** (64×64 grid)

| N samples | NS solver | FNO2d | Speedup |
|---|---|---|---|
| 1 | 29 ms | 4 ms | 6.5× |
| 10 | 283 ms | 33 ms | 8.6× |
| 50 | 1349 ms | 271 ms | **~5×** |

---

## How it works

### The problem

Solving a PDE classically requires numerical integration for every new initial condition. For 1000 samples this means 1000 independent solver runs — expensive in any design loop or simulation pipeline.

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
├── solvers/        # Classical PDE solvers
│   ├── burgers_1d.py          # Finite-difference solver for 1D Burgers
│   └── navier_stokes_2d.py    # Pseudo-spectral solver for 2D NS
├── data/           # Dataset generation, loading, normalisation
├── models/         # Neural architectures
│   ├── mlp.py                 # MLP baseline
│   ├── fno1d.py               # 1D Fourier Neural Operator
│   └── fno2d.py               # 2D Fourier Neural Operator
└── training/       # Trainer, metrics, config

scripts/
├── generate_data.py           # Generate 1D Burgers dataset
├── generate_data_2d.py        # Generate 2D Navier-Stokes dataset
├── train.py                   # Train MLP or FNO1d
├── train_2d.py                # Train FNO2d
├── benchmark_resolution.py    # Resolution generalization benchmark
├── benchmark_speed.py         # 1D speed benchmark
├── benchmark_2d.py            # 2D accuracy and speed benchmark
└── plot_loss.py               # Plot training curves

tests/                         # Mirrors src/ structure, 61 tests
```

---

## Quickstart

**Install**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**1D Burgers**

```bash
python scripts/generate_data.py
python scripts/train.py --model fno --mixed-resolution
python scripts/benchmark_resolution.py
python scripts/benchmark_speed.py
```

**2D Navier-Stokes**

```bash
python scripts/generate_data_2d.py --n-samples 1000 --output data/ns_64_1000.npz
python scripts/train_2d.py --data data/ns_64_1000.npz
python scripts/benchmark_2d.py --data data/ns_64_1000.npz
```

**Run tests**

```bash
python -m pytest
```

---

## Key implementation details

**Spectral convolution** (`src/fno/models/fno1d.py`, `fno2d.py`): The core FNO operation. Applies FFT, multiplies the lowest `n_modes` Fourier coefficients by learned complex weights, then applies inverse FFT. Weights are stored as real tensors and viewed as complex via `torch.view_as_complex`.

**Dynamic spatial grid**: The positional grid is computed from the actual input size at each forward pass — not stored as a fixed buffer. This is what enables resolution invariance without any model modification at inference time.

**Unit Gaussian normalisation** (`src/fno/data/transforms.py`): Mixed-resolution training uses per-sample normalisation (zero mean, unit std per sample) rather than dataset-level statistics. This is resolution-independent and consistent between training and evaluation.

**Pseudo-spectral NS solver** (`src/fno/solvers/navier_stokes_2d.py`): Computes spatial derivatives exactly in frequency space. Velocity is recovered from vorticity via the Poisson equation (trivial in Fourier space). Uses adaptive RK4 timesteps with a CFL-based stability condition and 2/3 dealiasing.

**Resolution batch sampler** (`scripts/train.py`): Ensures each batch contains samples from one resolution only, since tensors in a batch must share the same shape.

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- NumPy, SciPy, Matplotlib
