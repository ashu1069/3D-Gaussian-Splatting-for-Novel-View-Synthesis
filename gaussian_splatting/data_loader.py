"""
Data loading utilities for training 3D Gaussian Splatting.

This module handles loading training images, camera parameters, and point clouds
from COLMAP or similar structure-from-motion outputs.
"""

import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path


def load_image(image_path):
    """
    Load an image and convert to tensor.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image tensor. Shape: [H, W, 3], values in [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array)


def load_camera_parameters(cam_meta_path):
    """
    Load camera parameters from numpy file.
    
    Expected format: dict with keys:
    - 'fx', 'fy': Focal lengths
    - 'cx', 'cy': Principal point (or will be computed as image center)
    - 'height', 'width': Image dimensions
    - 'c2w': Camera-to-world matrices (optional, if per-image)
    
    Args:
        cam_meta_path: Path to camera metadata file
        
    Returns:
        Dictionary of camera parameters
    """
    cam_params = np.load(cam_meta_path, allow_pickle=True).item()
    return cam_params


def load_point_cloud(pcd_path):
    """
    Load point cloud from file.
    
    Supports various formats:
    - .ply files (ASCII or binary)
    - .npy files (numpy array)
    - .pt files (PyTorch tensor)
    
    Args:
        pcd_path: Path to point cloud file
        
    Returns:
        Points tensor. Shape: [N, 3]
    """
    ext = Path(pcd_path).suffix.lower()
    
    if ext == '.pt':
        return torch.load(pcd_path)
    elif ext == '.npy':
        points = np.load(pcd_path)
        return torch.from_numpy(points).float()
    elif ext == '.ply':
        return _load_ply(pcd_path)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")


def _load_ply(ply_path):
    """
    Load PLY file (simple ASCII format).
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        Points tensor. Shape: [N, 3]
    """
    points = []
    with open(ply_path, 'r') as f:
        header = True
        num_vertices = 0
        for line in f:
            if header:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('end_header'):
                    header = False
                    continue
            else:
                if len(points) < num_vertices:
                    coords = list(map(float, line.split()[:3]))
                    points.append(coords)
    
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Filter out NaN and Inf values
    valid_mask = torch.isfinite(points_tensor).all(dim=1)
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum().item()
        print(f"Warning: Filtering out {num_invalid} invalid points (NaN/Inf) from point cloud")
        points_tensor = points_tensor[valid_mask]
    
    # Filter out points that are extremely far away (likely corrupted)
    # Use a more robust approach: filter based on absolute bounds first
    if len(points_tensor) > 0:
        # First, filter by absolute bounds (reasonable scene size: ±1000 units)
        # This handles cases where all points are corrupted
        abs_mask = (torch.abs(points_tensor) < 1000).all(dim=1)
        
        if abs_mask.sum() > 0:
            # If we have points within reasonable bounds, use them
            points_tensor = points_tensor[abs_mask]
            print(f"Filtered to {len(points_tensor)} points within ±1000 unit bounds")
        else:
            # If all points are outside bounds, try a more lenient approach
            # Use percentile-based filtering instead
            center = points_tensor.mean(dim=0)
            distances = torch.norm(points_tensor - center, dim=1)
            # Keep points within 99th percentile (keeps 99% of points)
            if len(distances) > 100:
                percentile_99 = torch.quantile(distances, 0.99)
                reasonable_mask = distances < percentile_99
                if reasonable_mask.sum() > 0:
                    num_outliers = (~reasonable_mask).sum().item()
                    print(f"Warning: Using percentile filtering, keeping {reasonable_mask.sum()} points (removed {num_outliers} extreme outliers)")
                    points_tensor = points_tensor[reasonable_mask]
                else:
                    # Last resort: keep points within 100x median (very lenient)
                    median_dist = torch.median(distances)
                    if torch.isfinite(median_dist) and median_dist > 0:
                        max_dist = median_dist * 100
                        reasonable_mask = distances < max_dist
                        if reasonable_mask.sum() > 0:
                            points_tensor = points_tensor[reasonable_mask]
                            print(f"Warning: Using very lenient filtering, kept {len(points_tensor)} points")
    
    if len(points_tensor) == 0:
        raise ValueError("Point cloud is empty after filtering invalid values!")
    
    return points_tensor


class GaussianDataset:
    """
    Dataset class for loading training images and camera parameters.
    
    This class handles loading images, camera poses, and intrinsics for training.
    It assumes a directory structure with images and camera metadata.
    """
    
    def __init__(self, data_dir, image_dir='images', cam_meta_path=None,
                 scale_factor=0.5):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing data
            image_dir: Subdirectory containing images (relative to data_dir)
            cam_meta_path: Path to camera metadata file (or None to use default)
            scale_factor: Factor to scale images (1.0 = original size)
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / image_dir
        self.scale_factor = scale_factor
        
        # Load camera metadata
        if cam_meta_path is None:
            cam_meta_path = self.data_dir / 'cam_meta.npy'
        else:
            cam_meta_path = Path(cam_meta_path)
        
        self.cam_params = load_camera_parameters(cam_meta_path)
        
        # Get list of images
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')) +
                                  list(self.image_dir.glob('*.JPG')) +
                                  list(self.image_dir.glob('*.PNG')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Load camera poses if available
        self.c2w_matrices = self._load_camera_poses()
        
    def _load_camera_poses(self):
        """
        Load camera-to-world transformation matrices.
        
        Returns:
            List of c2w matrices, or None if not available
        """
        pose_file = self.data_dir / 'poses.npy'
        if pose_file.exists():
            poses = np.load(pose_file)
            return [torch.from_numpy(p).float() for p in poses]
        
        # Check if poses are in cam_params
        if 'c2w' in self.cam_params:
            if isinstance(self.cam_params['c2w'], list):
                return [torch.from_numpy(p).float() for p in self.cam_params['c2w']]
            elif isinstance(self.cam_params['c2w'], np.ndarray):
                return [torch.from_numpy(p).float() 
                        for p in self.cam_params['c2w']]
        
        return None
    
    def __len__(self):
        """Return number of images in dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with:
            - 'image': Image tensor [H, W, 3]
            - 'c2w': Camera-to-world matrix [4, 4]
            - 'fx', 'fy', 'cx', 'cy': Camera intrinsics
            - 'H', 'W': Image dimensions
        """
        # Load image
        image_path = self.image_files[idx]
        image = load_image(image_path)
        
        # Scale image if needed
        if self.scale_factor != 1.0:
            H, W = image.shape[:2]
            new_H, new_W = int(H * self.scale_factor), int(W * self.scale_factor)
            image = torch.nn.functional.interpolate(
                image.permute(2, 0, 1).unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        
        H, W = image.shape[:2]
        
        # Get camera pose
        if self.c2w_matrices is not None:
            c2w = self.c2w_matrices[idx]
            if not isinstance(c2w, torch.Tensor):
                c2w = torch.from_numpy(c2w).float()
        else:
            # Default: identity (camera at origin looking down -z)
            c2w = torch.eye(4, dtype=torch.float32)
        
        # Get intrinsics
        fx = self.cam_params['fx'] * self.scale_factor
        fy = self.cam_params['fy'] * self.scale_factor
        
        # Principal point defaults to image center if not specified
        if 'cx' in self.cam_params and 'cy' in self.cam_params:
            cx = self.cam_params['cx'] * self.scale_factor
            cy = self.cam_params['cy'] * self.scale_factor
        else:
            cx = W / 2.0
            cy = H / 2.0
        
        return {
            'image': image,
            'c2w': c2w,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'H': H,
            'W': W,
            'idx': idx
        }


def initialize_gaussians_from_pointcloud(points, num_sh_bands=3):
    """
    Initialize 3D Gaussians from a point cloud.
    
    We initialize Gaussians at each point in the point cloud with:
    - Position: Point location
    - Scale: Small initial size (learned during training)
    - Rotation: Identity (no rotation initially)
    - Opacity: High initial opacity (learned during training)
    - Color: Random or from point cloud colors (if available)
    
    The number of SH bands determines view-dependent color complexity:
    - bands=0: View-independent color only
    - bands=1: Linear view dependence
    - bands=3: Full view dependence (standard)
    
    Args:
        points: Point cloud. Shape: [N, 3] or [N, 6] (if includes RGB)
        num_sh_bands: Number of spherical harmonic bands (0-3)
        
    Returns:
        Dictionary with initialized Gaussian parameters:
        - 'pos': Positions [N, 3]
        - 'opacity_raw': Raw opacity values [N]
        - 'f_dc': DC SH term [N, 3]
        - 'f_rest': Higher-order SH terms [N, 45] (if bands=3)
        - 'scale_raw': Log-scale parameters [N, 3]
        - 'q_raw': Quaternion rotations [N, 4]
    """
    N = points.shape[0]
    device = points.device if isinstance(points, torch.Tensor) else 'cpu'
    
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).float()
    
    # Extract positions (first 3 columns)
    pos = points[:, :3].to(device)
    
    # Initialize scales: small random values in log space
    # Exponentiating gives scales around 0.01-0.1
    scale_raw = torch.randn(N, 3, device=device) * 0.1 - 2.0
    
    # Initialize rotations: identity quaternions [0, 0, 0, 1]
    q_raw = torch.zeros(N, 4, device=device)
    q_raw[:, 3] = 1.0  # w component = 1 (no rotation)
    
    # Initialize opacity: high initial value (sigmoid(0) = 0.5)
    # We use a small positive value so sigmoid gives ~0.5-0.7
    opacity_raw = torch.ones(N, device=device) * 0.1
    
    # Initialize colors
    if points.shape[1] >= 6:
        # Use colors from point cloud
        colors = points[:, 3:6].to(device)
        # Convert to [0, 1] if in [0, 255]
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        # Random colors
        colors = torch.rand(N, 3, device=device)
    
    # Initialize SH coefficients
    # DC term (l=0): view-independent color
    f_dc = colors  # [N, 3]
    
    # Higher-order terms (l=1,2,3): start near zero (learned during training)
    if num_sh_bands >= 3:
        f_rest = torch.zeros(N, 45, device=device)  # 15 terms × 3 channels
    elif num_sh_bands == 1:
        f_rest = torch.zeros(N, 9, device=device)  # 3 terms × 3 channels
    else:
        f_rest = torch.zeros(N, 0, device=device)
    
    return {
        'pos': pos,
        'opacity_raw': opacity_raw,
        'f_dc': f_dc,
        'f_rest': f_rest,
        'scale_raw': scale_raw,
        'q_raw': q_raw
    }

