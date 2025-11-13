# 3D Gaussian Splatting: From Scratch Implementation

A complete implementation of 3D Gaussian Splatting from scratch in PyTorch. This project provides a full rendering pipeline for novel view synthesis using 3D Gaussians as a scene representation, including training, inference, and rendering capabilities.

**Current Status**: Pure PyTorch implementation achieving ~0.2-1 FPS. **Contributions welcome** for CUDA kernel optimization to reach 100+ FPS performance!

## Overview

3D Gaussian Splatting is a state-of-the-art technique for novel view synthesis that represents scenes as a collection of 3D Gaussians. Each Gaussian has:
- **Position** (μ): 3D location in space
- **Covariance** (Σ): Shape and orientation (via scale + rotation)
- **Opacity** (α): Transparency control
- **Color**: View-dependent appearance via spherical harmonics

The method achieves high-quality rendering by projecting 3D Gaussians onto the 2D image plane and compositing them using alpha blending.

## Features

- **Complete Implementation**: Full 3D Gaussian Splatting pipeline from scratch
- **Comprehensive Documentation**: Detailed theoretical explanations for every component
- **Modular Design**: Clean, well-organized codebase with clear separation of concerns
- **Complete Pipeline**: Data loading, training, inference, and rendering
- **Pure PyTorch**: Current implementation uses PyTorch operations (no custom CUDA kernels yet)
- **View-Dependent Appearance**: Spherical harmonics for realistic material rendering
- **Easy to Use**: Simple API for training and rendering novel views

## Performance & Optimization

**Current Performance**: ~0.2-1 FPS (pure PyTorch implementation)

**Optimization Opportunities**:
- CUDA kernel implementation for rasterization
- Tile-based culling optimization
- Memory-efficient Gaussian sorting
- Parallel projection and blending
- Integration with existing CUDA libraries (gsplat, diff-gaussian-rasterization)

**Contributions Welcome!** If you're interested in optimizing this implementation, please open an issue or submit a PR. Areas that need optimization:
1. CUDA rasterization kernels for faster rendering
2. Memory optimization for large scenes
3. Multi-GPU rendering support
4. Performance profiling and benchmarking tools

## Requirements

- Python 3.7+
- PyTorch (with CUDA support recommended)
- NumPy
- Pillow (PIL)
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fast-Gaussian-Splatting-for-Novel-View-Synthesis
```

2. Install dependencies:
```bash
pip install torch numpy pillow tqdm
```

Or create a `requirements.txt`:
```txt
torch>=1.9.0
numpy>=1.21.0
Pillow>=8.0.0
tqdm>=4.62.0
```

Then install:
```bash
pip install -r requirements.txt
```

## Project Structure

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

### Module Descriptions

**Core Modules (`gaussian_splatting/`):**
- **`gaussian.py`**: Handles Gaussian parameterization, including quaternion-to-rotation conversion and covariance matrix construction
- **`spherical_harmonics.py`**: Implements spherical harmonics evaluation for view-dependent appearance
- **`render.py`**: Core rendering pipeline implementing tile-based rasterization and alpha blending
- **`utils.py`**: Geometric utilities for point projection, matrix operations, and camera intrinsics scaling
- **`losses.py`**: Loss functions (L1 and SSIM) used during training
- **`data_loader.py`**: Data loading utilities, including dataset class and point cloud initialization

**Scripts (`scripts/`):**
- **`train.py`**: Main training script with adaptive density control
- **`render_trained.py`**: User-friendly rendering script with automatic orbit generation and video creation
- **`inference.py`**: Advanced inference script for custom camera trajectories

**Dataset Utilities (`datasets/`):**
- **`download_mipnerf360.py`**: Downloads Mip-NeRF 360 dataset scenes
- **`prepare_mipnerf360.py`**: Converts Mip-NeRF 360 format to training format
- **`setup_dataset.sh`**: Automated script combining download and preparation

**Note**: All scripts should be run from the repository root directory.

## Quick Start

### Dataset Setup (Mip-NeRF 360)

We provide scripts to download and prepare the Mip-NeRF 360 dataset:

**Option 1: Automated Setup (Recommended)**
```bash
# Download and prepare a scene (e.g., 'garden')
./datasets/setup_dataset.sh garden

# Or for other scenes: bicycle, bonsai, counter, kitchen, room, stump, flowers, treehill
./datasets/setup_dataset.sh bicycle
```

**Option 2: Manual Setup**
```bash
# Step 1: Download a scene
python datasets/download_mipnerf360.py --scene garden --download_dir data/mipnerf360

# Step 2: Prepare for training
python datasets/prepare_mipnerf360.py \
    --input_dir data/mipnerf360/garden \
    --output_dir data/prepared/garden \
    --scene_name garden
```

**Option 3: Manual Download (If automated download fails)**

If the automatic download fails (404 error), download manually:

1. **Main scenes** (garden, bicycle, bonsai, counter, kitchen, room, stump):
   ```bash
   # Download from Google Cloud Storage
   wget https://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O data/mipnerf360/360_v2.zip
   unzip data/mipnerf360/360_v2.zip -d data/mipnerf360/
   ```

2. **Extra scenes** (flowers, treehill):
   ```bash
   wget https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip -O data/mipnerf360/360_extra_scenes.zip
   unzip data/mipnerf360/360_extra_scenes.zip -d data/mipnerf360/
   ```

3. **Prepare for training:**
   ```bash
   python prepare_mipnerf360.py \
       --input_dir data/mipnerf360/garden \
       --output_dir data/prepared/garden \
       --scene_name garden
   ```

**Available Scenes:**
- `garden` - Garden scene (recommended for first try, ~2GB)
- `bicycle` - Bicycle scene (~2GB)
- `bonsai` - Bonsai tree (~1GB)
- `counter` - Kitchen counter (~1GB)
- `kitchen` - Kitchen scene (~2GB)
- `room` - Room interior (~1GB)
- `stump` - Tree stump (~1GB)
- `flowers` - Flower field (~2GB, requires extra scenes zip)
- `treehill` - Tree on hill (~2GB, requires extra scenes zip)

**Dataset Structure After Preparation:**
```
data/prepared/garden/
├── images/              # Training images (000000.jpg, 000001.jpg, ...)
├── cam_meta.npy        # Camera intrinsics
├── poses.npy           # Camera poses [N, 4, 4]
└── pointcloud.ply      # Initial point cloud (optional, from COLMAP)
```

### Training

Train 3D Gaussians from images:

```bash
python scripts/train.py \
    --data_dir data/prepared/garden \
    --output_dir output/garden \
    --iterations 7000 \
    --batch_size 2 \
    --scale_factor 0.5
```

**Training Parameters:**
- `--data_dir`: Directory containing prepared training data
- `--output_dir`: Directory to save checkpoints
- `--iterations`: Number of training iterations (7000 recommended, 30000 for full quality)
- `--batch_size`: Number of views per iteration (default: 1)
- `--scale_factor`: Image scale factor to reduce memory (0.5 = half resolution, default: 0.5)
- `--resume_from`: Resume from checkpoint (e.g., `output/garden/checkpoint_final.pt`)
- `--pcd_path`: Path to initial point cloud (optional, auto-detected if in data_dir)

**Training Tips:**
- **Iterations**: Start with 7000 iterations for quick results, use 30000 for publication-quality
- **Memory**: If OOM errors occur, reduce `--scale_factor` to 0.25 or use smaller batch size
- **Multi-GPU**: Automatically detected and used if available
- **Checkpoints**: Saved every 1000 iterations by default

The training data directory should contain:
- `images/`: Directory with training images
- `cam_meta.npy`: Camera metadata file (see File Format Expectations)
- `poses.npy` (optional): Camera poses for each image
- `pointcloud.ply` (optional): Initial point cloud from COLMAP (auto-detected)

### Inference / Rendering

After training, render novel views using the user-friendly script:

```bash
python render_trained.py \
    --checkpoint_dir output/garden \
    --data_dir data/prepared/garden \
    --iteration final \
    --output_dir renders/garden \
    --orbit_frames 60 \
    --scale_factor 0.5 \
    --video_fps 30
```

**Rendering Parameters:**
- `--checkpoint_dir`: Directory containing trained checkpoints
- `--data_dir`: Directory with training data (for camera intrinsics)
- `--iteration`: Checkpoint iteration to use (`final` or `000000`, `001000`, etc.)
- `--output_dir`: Directory to save rendered images and video
- `--orbit_frames`: Number of frames for orbit video (default: 60)
- `--scale_factor`: Render resolution scale (0.5 = half resolution, default: 1.0)
- `--video_fps`: FPS for output video (default: 30)
- `--render_training_views`: Also render training views for comparison
- `--render_orbit`: Render circular orbit around scene (default: True)
- `--benchmark_only`: Only measure FPS without saving images

**Advanced Usage (Programmatic):**

For custom camera trajectories, use `inference.py`:

```python
from scripts.inference import render_novel_views

render_novel_views(
    pos_path='output/garden/pos_final.pt',
    opacity_raw_path='output/garden/opacity_raw_final.pt',
    f_dc_path='output/garden/f_dc_final.pt',
    f_rest_path='output/garden/f_rest_final.pt',
    scale_raw_path='output/garden/scale_raw_final.pt',
    q_raw_path='output/garden/q_rot_final.pt',
    cam_meta_path='data/prepared/garden/cam_meta.npy',
    camera_trajectory_path='trajectories/orbit.pt',
    output_dir='novel_views',
    scale_factor=0.5
)
```

### Using Individual Components

You can also use the components independently:

```python
import torch
from gaussian_splatting.gaussian import build_sigma_from_params, quat_to_rotmat
from gaussian_splatting.spherical_harmonics import evaluate_sh
from gaussian_splatting.render import render
from gaussian_splatting.utils import project_points, scale_intrinsics

# Build covariance matrices from parameters
scale_raw = torch.load('scale_raw.pt')
q_raw = torch.load('q_rot.pt')
sigma = build_sigma_from_params(scale_raw, q_raw)

# Evaluate view-dependent colors
f_dc = torch.load('f_dc.pt')
f_rest = torch.load('f_rest.pt')
points = torch.load('pos.pt')
c2w = torch.eye(4)  # Your camera pose
colors = evaluate_sh(f_dc, f_rest, points, c2w)

# Render
pos = torch.load('pos.pt')
opacity_raw = torch.load('opacity_raw.pt')
img = render(pos, colors, opacity_raw, sigma, c2w, H=512, W=512, 
             fx=500, fy=500, cx=256, cy=256)
```

## Theory Overview

### 3D Gaussian Representation

Each Gaussian is defined by:
```
G(x) = exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))
```

The covariance matrix Σ is decomposed as:
```
Σ = R S Sᵀ Rᵀ
```
where R is a rotation matrix (from quaternion) and S is a diagonal scale matrix.

### Rendering Pipeline

1. **Projection**: Transform 3D Gaussians to camera space and project to 2D
2. **Covariance Projection**: Project 3D covariance to 2D using the Jacobian:
   ```
   Σ₂D = J @ Σ₃D_camera @ Jᵀ
   ```
3. **Tiling**: Divide image into tiles for efficient parallel processing
4. **Rasterization**: Evaluate Gaussian contributions per pixel
5. **Alpha Blending**: Composite Gaussians front-to-back:
   ```
   C_final = Σᵢ αᵢ * cᵢ * Πⱼ<ᵢ (1 - αⱼ)
   ```

### Spherical Harmonics

View-dependent colors are computed using spherical harmonics:
```
color(view_dir) = sigmoid(Σₖ fₖ * Yₖ(view_dir))
```

We use 16 SH basis functions (degrees 0-3) to capture view-dependent effects like specular highlights and reflections.

### Training Process

Training optimizes Gaussian parameters to minimize reconstruction error:

1. **Loss Function**: Combined L1 + SSIM loss
   ```
   L = λ_L1 * L_L1 + λ_SSIM * L_SSIM
   ```
   where λ_L1 ≈ 0.8 and λ_SSIM ≈ 0.2

2. **Optimization**: Different learning rates for different parameters
   - Positions: High LR (0.00016) - needs to move quickly
   - Opacity: Medium LR (0.05) - controls visibility
   - Scale: High LR (0.005) - controls size
   - Rotation: Low LR (0.001) - fine-grained orientation
   - SH coefficients: Different LR for DC vs rest terms

3. **Adaptive Density Control**: Periodically adjust Gaussian count
   - **Split**: Large Gaussians with high gradients → refine details
   - **Clone**: Small Gaussians with high gradients → fill gaps
   - **Prune**: Low-opacity Gaussians → remove unused ones

## API Documentation

### `gaussian.py`

- **`quat_to_rotmat(quat)`**: Convert quaternions to rotation matrices
- **`build_sigma_from_params(scale_raw, q_raw)`**: Build 3D covariance matrices from scale and rotation parameters

### `spherical_harmonics.py`

- **`evaluate_sh(f_dc, f_rest, points, c2w)`**: Evaluate spherical harmonics to compute view-dependent colors
- **`HARMONICS`**: Dictionary of precomputed SH normalization constants

### `render.py`

- **`render(pos, color, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy, ...)`**: Main rendering function
  - `pos`: 3D positions [N, 3]
  - `color`: RGB colors [N, 3]
  - `opacity_raw`: Raw opacity values [N]
  - `sigma`: 3D covariance matrices [N, 3, 3]
  - `c2w`: Camera-to-world transform [4, 4]
  - Returns: Rendered image [H, W, 3]

### `utils.py`

- **`project_points(pc, c2w, fx, fy, cx, cy)`**: Project 3D points to 2D image coordinates
- **`inv2x2(M, eps=1e-12)`**: Efficient 2x2 matrix inversion
- **`scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy)`**: Scale camera intrinsics for different resolutions

### `losses.py`

- **`l1_loss(pred, target)`**: Compute L1 (mean absolute error) loss
- **`ssim_loss(pred, target)`**: Compute SSIM (Structural Similarity Index) loss
- **`compute_loss(pred, target, lambda_l1, lambda_ssim)`**: Combined L1 + SSIM loss

### `data_loader.py`

- **`GaussianDataset`**: Dataset class for loading training images and camera parameters
- **`initialize_gaussians_from_pointcloud(points, num_sh_bands)`**: Initialize Gaussians from point cloud
- **`load_point_cloud(pcd_path)`**: Load point cloud from various formats (.ply, .npy, .pt)

### `train.py`

- **`train(...)`**: Main training function
- **`GaussianModel`**: Model class managing learnable Gaussian parameters

### `inference.py`

- **`render_novel_views(...)`**: High-level function to render a sequence of novel views from trained Gaussians

## 🔧 Usage Examples

### Example 1: Render Single View

```python
import torch
from gaussian_splatting.render import render
from gaussian_splatting.gaussian import build_sigma_from_params
from gaussian_splatting.spherical_harmonics import evaluate_sh

# Load trained parameters
pos = torch.load('pos.pt').cuda()
opacity_raw = torch.load('opacity_raw.pt').cuda()
f_dc = torch.load('f_dc.pt').cuda()
f_rest = torch.load('f_rest.pt').cuda()
scale_raw = torch.load('scale_raw.pt').cuda()
q_raw = torch.load('q_rot.pt').cuda()

# Build covariance
sigma = build_sigma_from_params(scale_raw, q_raw)

# Set up camera
c2w = torch.eye(4).cuda()  # Your camera pose
H, W = 512, 512
fx, fy, cx, cy = 500, 500, 256, 256

# Evaluate colors
colors = evaluate_sh(f_dc, f_rest, pos, c2w)

# Render
with torch.no_grad():
    img = render(pos, colors, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy)
    
# Save
from PIL import Image
import numpy as np
Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save('output.png')
```

### Example 3: Custom Camera Trajectory

```python
import torch
from scripts.inference import render_novel_views

# Create circular orbit trajectory
n_frames = 60
radius = 3.0
c2ws = []
for i in range(n_frames):
    angle = 2 * torch.pi * i / n_frames
    c2w = torch.eye(4)
    c2w[0, 3] = radius * torch.cos(angle)
    c2w[2, 3] = radius * torch.sin(angle)
    # Look at origin
    c2w[:3, :3] = torch.tensor([
        [-torch.sin(angle), 0, -torch.cos(angle)],
        [0, 1, 0],
        [torch.cos(angle), 0, -torch.sin(angle)]
    ])
    c2ws.append(c2w)
torch.save(torch.stack(c2ws), 'orbit.pt')

# Render
render_novel_views(
    pos_path='pos.pt',
    opacity_raw_path='opacity_raw.pt',
    f_dc_path='f_dc.pt',
    f_rest_path='f_rest.pt',
    scale_raw_path='scale_raw.pt',
    q_raw_path='q_rot.pt',
    cam_meta_path='cam_meta.npy',
    camera_trajectory_path='orbit.pt',
    output_dir='orbit_views'
)
```

## Rendering Parameters

The `render()` function accepts several parameters for fine-tuning:

- **`near`, `far`**: Depth clipping planes (default: 2e-3, 100)
- **`pix_guard`**: Pixel guard band for culling (default: 64)
- **`T`**: Tile size for rasterization (default: 16)
- **`chi_square_clip`**: Chi-square clipping threshold (default: 9.21 ≈ 3σ)
- **`alpha_max`**: Maximum alpha value (default: 0.99)
- **`alpha_cutoff`**: Alpha cutoff threshold (default: 1/255)

## Troubleshooting

### Dataset Setup Issues

**No point cloud found:**
- If COLMAP point cloud is not available, training will initialize from random points
- This is fine, but training may take longer to converge
- Ensure `sparse/0/points3D.bin` exists in the dataset directory

**Download fails (404 error):**
- Use manual download option (see Option 3 in Dataset Setup)
- Check official website: https://jonbarron.info/mipnerf360/
- Some scenes (flowers, treehill) may require requesting access

**Image format issues:**
- The preparation script handles various formats (.jpg, .png, .JPG, .PNG)
- Ensure images are RGB format

**Memory issues during training:**
- Reduce image resolution using `--scale_factor 0.25`
- Reduce batch size: `--batch_size 1`
- Train on a smaller scene first

### Training Issues

**Out of Memory (OOM):**
- Use `--scale_factor 0.25` or `0.5` to reduce image resolution
- Reduce `--batch_size` to 1
- Clear GPU cache: `torch.cuda.empty_cache()`

**Black rendered images:**
- Ensure training completed successfully (check loss values)
- Verify camera poses are correct
- Check that Gaussian positions are valid (no NaN values)

**Slow training:**
- Use multiple GPUs (automatically detected)
- Increase `--batch_size` if memory allows
- Use `--scale_factor 0.5` for faster iterations

### Rendering Issues

**Checkpoint not found:**
- Check that checkpoint files exist in `--checkpoint_dir`
- Use `--iteration final` to automatically find latest checkpoint
- Verify checkpoint format matches expected structure

**Low FPS:**
- Current implementation uses pure PyTorch (no custom CUDA kernels)
- Performance: ~0.2-1 FPS (much slower than optimized implementations)
- To improve: Reduce `--scale_factor` (e.g., 0.25), use fewer frames, or train for fewer iterations
- **Optimization needed**: CUDA kernel implementation for rasterization (contributions welcome!)

## Testing

To verify the installation:

```python
python -c "from gaussian_splatting.gaussian import build_sigma_from_params; from gaussian_splatting.spherical_harmonics import evaluate_sh; from gaussian_splatting.render import render; print('✓ All imports successful!')"
```

## File Format Expectations

The inference script expects the following file formats:

- **Position** (`pos.pt`): `torch.Tensor` of shape `[N, 3]` (3D positions)
- **Opacity** (`opacity_raw.pt`): `torch.Tensor` of shape `[N]` (raw opacity values)
- **SH DC term** (`f_dc.pt`): `torch.Tensor` of shape `[N, 3]` (view-independent color)
- **SH rest** (`f_rest.pt`): `torch.Tensor` of shape `[N, 45]` (view-dependent coefficients)
- **Scale** (`scale_raw.pt`): `torch.Tensor` of shape `[N, 3]` (log-scale parameters)
- **Rotation** (`q_rot.pt`): `torch.Tensor` of shape `[N, 4]` (quaternion rotations)
- **Camera metadata** (`cam_meta.npy`): `dict` with keys `['fx', 'fy', 'height', 'width']`
- **Camera trajectory** (`orbit.pt`): `torch.Tensor` of shape `[K, 4, 4]` (camera-to-world transforms)

## Contributing

Contributions are welcome! This project is actively seeking optimizations and improvements:

### High Priority
- **CUDA Kernel Implementation**: Write custom CUDA kernels for rasterization to achieve 100+ FPS
- **Performance Optimization**: Memory efficiency, parallel processing, multi-GPU support
- **Benchmarking**: Performance profiling tools and benchmarks

### General Contributions
- Bug fixes and code improvements
- Additional features and enhancements
- Documentation improvements
- Test cases and examples

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/optimization`)
3. Make your changes
4. Submit a pull request with a clear description

**Areas Needing Optimization**:
- Rasterization CUDA kernels (highest impact on FPS)
- Memory-efficient Gaussian management
- Tile-based culling optimization
- Multi-GPU rendering pipeline
- Integration with existing CUDA libraries

Please feel free to open issues to discuss optimization strategies or submit PRs!

## Acknowledgments

This implementation is inspired by the original 3D Gaussian Splatting paper:
- Kerbl, B., Kopanas, G., Diolatzis, S., Drettakis, G., & Leimkühler, T. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.

We also acknowledge the following resources:
- [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) for providing benchmark scenes
- The open-source community for inspiration and feedback

## Citation
Original paper:
```bibtex
@article{kerbl2023gaussian,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Diolatzis, Stavros and Drettakis, George and Leimk{\"u}hler, Thomas},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

