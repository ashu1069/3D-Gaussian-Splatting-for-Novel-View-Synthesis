# Repository Structure

This document describes the organization of the repository.

## Directory Layout

```
Fast-Gaussian-Splatting-for-Novel-View-Synthesis/
├── gaussian_splatting/          # Core modules (mathematical components)
│   ├── __init__.py              # Package initialization and exports
│   ├── gaussian.py              # Gaussian parameterization (quaternions, covariance)
│   ├── spherical_harmonics.py   # Spherical harmonics for view-dependent colors
│   ├── render.py                # Main rendering pipeline (pure PyTorch)
│   ├── utils.py                 # Geometric utilities (projection, matrix ops)
│   ├── losses.py                # Loss functions (L1, SSIM)
│   └── data_loader.py           # Data loading utilities
│
├── scripts/                     # Training & inference scripts
│   ├── train.py                 # Main training script
│   ├── render_trained.py        # User-friendly rendering script (orbit, video)
│   └── inference.py             # Advanced inference script
│
├── datasets/                     # Dataset utilities
│   ├── download_mipnerf360.py   # Download Mip-NeRF 360 dataset
│   ├── prepare_mipnerf360.py    # Convert Mip-NeRF 360 to training format
│   └── setup_dataset.sh          # Automated dataset setup script
│
├── README.md                     # Main documentation
├── LICENSE                       # License file
├── requirements.txt              # Python dependencies
└── .gitignore                   # Git ignore rules
```

## Module Descriptions

### Core Modules (`gaussian_splatting/`)

- **`gaussian.py`**: Handles Gaussian parameterization, including quaternion-to-rotation conversion and covariance matrix construction
- **`spherical_harmonics.py`**: Implements spherical harmonics evaluation for view-dependent appearance
- **`render.py`**: Core rendering pipeline implementing tile-based rasterization and alpha blending
- **`utils.py`**: Geometric utilities for point projection, matrix operations, and camera intrinsics scaling
- **`losses.py`**: Loss functions (L1 and SSIM) used during training
- **`data_loader.py`**: Data loading utilities, including dataset class and point cloud initialization

### Scripts (`scripts/`)

- **`train.py`**: Main training script with adaptive density control
- **`render_trained.py`**: User-friendly rendering script with automatic orbit generation and video creation
- **`inference.py`**: Advanced inference script for custom camera trajectories

### Dataset Utilities (`datasets/`)

- **`download_mipnerf360.py`**: Downloads Mip-NeRF 360 dataset scenes
- **`prepare_mipnerf360.py`**: Converts Mip-NeRF 360 format to training format
- **`setup_dataset.sh`**: Automated script combining download and preparation

## Usage

All scripts should be run from the repository root:

```bash
# Training
python scripts/train.py --data_dir data/prepared/garden --output_dir output/garden

# Rendering
python scripts/render_trained.py --checkpoint_dir output/garden --data_dir data/prepared/garden

# Dataset setup
./datasets/setup_dataset.sh garden
```

## Import Structure

Core modules can be imported as:

```python
from gaussian_splatting.gaussian import build_sigma_from_params
from gaussian_splatting.spherical_harmonics import evaluate_sh
from gaussian_splatting.render import render
from gaussian_splatting.utils import project_points
from gaussian_splatting.losses import compute_loss
from gaussian_splatting.data_loader import GaussianDataset
```

Or using the package's `__init__.py`:

```python
from gaussian_splatting import build_sigma_from_params, evaluate_sh, render
```
