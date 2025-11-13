"""
Inference script for rendering novel views from trained 3D Gaussians.

NOTE: For most users, `render_trained.py` is recommended as it's more user-friendly
and automatically handles checkpoint loading and camera trajectory generation.

This script (`inference.py`) is useful when you have a pre-computed camera trajectory
file and want more control over the rendering process.

After training, 3D Gaussians have learned parameters that represent the scene:
- Positions (μ): Where Gaussians are located in 3D space
- Covariances (Σ): Shape and orientation of each Gaussian
- Opacities (α): How opaque each Gaussian is
- Spherical Harmonic coefficients: View-dependent appearance

To render a novel view:
1. Load trained parameters from disk
2. Build 3D covariance matrices from scale/quaternion parameters
3. For each camera pose in trajectory:
   a. Evaluate SH coefficients to get view-dependent colors
   b. Project Gaussians to 2D and render using alpha blending
   c. Save rendered image

This enables generating novel views of the scene from arbitrary camera positions,
demonstrating the scene's 3D understanding.
"""

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

try:
    from .gaussian import build_sigma_from_params
    from .spherical_harmonics import evaluate_sh
    from .render import render
    from .utils import scale_intrinsics
except ImportError:
    from gaussian import build_sigma_from_params
    from spherical_harmonics import evaluate_sh
    from render import render
    from utils import scale_intrinsics


def render_novel_views(
    pos_path,
    opacity_raw_path,
    f_dc_path,
    f_rest_path,
    scale_raw_path,
    q_raw_path,
    cam_meta_path,
    camera_trajectory_path,
    output_dir='novel_views',
    scale_factor=2
):
    """
    Render novel views from trained 3D Gaussians.
    
    Pipeline:
    ---------
    1. Load trained Gaussian parameters (positions, opacities, SH coefficients, scales, rotations)
    2. Reconstruct 3D covariance matrices from scale/quaternion parameters
    3. Load camera intrinsics and trajectory (sequence of camera poses)
    4. For each camera pose:
       a. Scale intrinsics to target resolution
       b. Evaluate spherical harmonics to get view-dependent colors
       c. Render image using 3D Gaussian Splatting
       d. Save rendered image
    
    The scale_factor parameter allows rendering at lower resolution for faster inference
    or higher resolution for better quality. The intrinsics are scaled proportionally
    to maintain the camera's field of view.
    
    Args:
        pos_path: Path to position tensor file. Shape: [N, 3]
        opacity_raw_path: Path to opacity_raw tensor file. Shape: [N]
        f_dc_path: Path to f_dc (DC SH term) tensor file. Shape: [N, 3]
        f_rest_path: Path to f_rest (higher-order SH terms) tensor file. Shape: [N, 45]
        scale_raw_path: Path to scale_raw (log-scale) tensor file. Shape: [N, 3]
        q_raw_path: Path to quaternion rotation tensor file. Shape: [N, 4]
        cam_meta_path: Path to camera metadata numpy file (contains fx, fy, height, width)
        camera_trajectory_path: Path to camera trajectory tensor file. Shape: [K, 4, 4]
                               Each [4,4] matrix is a camera-to-world transform
        output_dir: Directory to save rendered images
        scale_factor: Factor to scale down resolution (e.g., 2 means half resolution)
                     This reduces computation while maintaining field of view
    """
    # Load trained Gaussian parameters
    pos = torch.load(pos_path).cuda()
    opacity_raw = torch.load(opacity_raw_path).cuda()
    f_dc = torch.load(f_dc_path).cuda()
    f_rest = torch.load(f_rest_path).cuda()
    scale_raw = torch.load(scale_raw_path).cuda()
    q_raw = torch.load(q_raw_path).cuda()

    # Load camera parameters and trajectory
    cam_parameters = np.load(cam_meta_path, allow_pickle=True).item()
    orbit_c2ws = torch.load(camera_trajectory_path).cuda()

    # Build covariance matrices
    sigma = build_sigma_from_params(scale_raw, q_raw)

    # Render each view (disable gradients for inference)
    with torch.no_grad():
        for i, c2w_i in tqdm(enumerate(orbit_c2ws)):
            c2w = c2w_i
            H = cam_parameters['height'] // scale_factor
            W = cam_parameters['width'] // scale_factor
            H_src = cam_parameters['height']
            W_src = cam_parameters['width']
            fx, fy = cam_parameters['fx'], cam_parameters['fy']
            cx, cy = W_src / 2, H_src / 2
            fx, fy, cx, cy = scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy)

            # Evaluate spherical harmonics for view-dependent color
            color = evaluate_sh(f_dc, f_rest, pos, c2w)
            
            # Render image
            img = render(pos, color, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy)

            # Save image
            Image.fromarray((img.cpu().detach().numpy() * 255).astype(np.uint8)).save(
                f'{output_dir}/frame_{i:04d}.png'
            )


if __name__ == "__main__":
    """
    Example usage - requires pre-computed camera trajectory file.
    
    For easier usage, see render_trained.py which:
    - Automatically loads checkpoints
    - Generates camera trajectories automatically
    - Works directly with training output structure
    
    Example with this script:
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Render novel views with pre-computed camera trajectory',
        epilog='NOTE: For easier usage, consider using render_trained.py instead'
    )
    parser.add_argument('--pos_path', type=str, required=True)
    parser.add_argument('--opacity_raw_path', type=str, required=True)
    parser.add_argument('--f_dc_path', type=str, required=True)
    parser.add_argument('--f_rest_path', type=str, required=True)
    parser.add_argument('--scale_raw_path', type=str, required=True)
    parser.add_argument('--q_raw_path', type=str, required=True)
    parser.add_argument('--cam_meta_path', type=str, required=True)
    parser.add_argument('--camera_trajectory_path', type=str, required=True,
                        help='Path to .pt file containing camera trajectory [K, 4, 4]')
    parser.add_argument('--output_dir', type=str, default='novel_views')
    parser.add_argument('--scale_factor', type=int, default=2,
                        help='Scale factor (2 = half resolution)')
    
    args = parser.parse_args()
    
    render_novel_views(
        pos_path=args.pos_path,
        opacity_raw_path=args.opacity_raw_path,
        f_dc_path=args.f_dc_path,
        f_rest_path=args.f_rest_path,
        scale_raw_path=args.scale_raw_path,
        q_raw_path=args.q_raw_path,
        cam_meta_path=args.cam_meta_path,
        camera_trajectory_path=args.camera_trajectory_path,
        output_dir=args.output_dir,
        scale_factor=args.scale_factor
    )

