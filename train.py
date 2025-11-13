"""
Training script for 3D Gaussian Splatting.

Training 3D Gaussians involves optimizing Gaussian parameters to minimize
reconstruction error between rendered and ground truth images. Key components:

1. Forward Pass: Render image using current Gaussian parameters
2. Loss Computation: Compare rendered image to ground truth (L1 + SSIM)
3. Backward Pass: Compute gradients and update parameters
4. Adaptive Density Control: Periodically split/prune/clone Gaussians based on gradients

The optimization uses different learning rates for different parameters:
- Position: High LR (0.00016) - needs to move quickly
- Opacity: Medium LR (0.05) - controls visibility
- Scale: High LR (0.005) - controls size
- Rotation: Low LR (0.001) - fine-grained orientation
- SH coefficients: Different LR for DC vs rest terms

Adaptive density control:
- Split: Large Gaussians with high gradient magnitude
- Clone: Small Gaussians with high gradient magnitude
- Prune: Low-opacity Gaussians (< threshold)

This ensures Gaussians are placed where needed and removed where not needed.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import json
import numpy as np

from gaussian import build_sigma_from_params
from spherical_harmonics import evaluate_sh
from render import render
from losses import compute_loss
from data_loader import GaussianDataset, initialize_gaussians_from_pointcloud, load_point_cloud


class GaussianModel:
    """
    Model class for 3D Gaussians with learnable parameters.
    
    This class manages all Gaussian parameters as learnable tensors and provides
    methods for optimization and adaptive density control.
    """
    
    def __init__(self, initial_params, device='cuda'):
        """
        Initialize Gaussian model with parameters.
        
        Args:
            initial_params: Dictionary with initial parameters (from initialize_gaussians_from_pointcloud)
            device: Device to store parameters on
        """
        self.device = device
        
        # Convert all parameters to learnable tensors
        self.pos = torch.nn.Parameter(initial_params['pos'].to(device))
        self.opacity_raw = torch.nn.Parameter(initial_params['opacity_raw'].to(device))
        self.f_dc = torch.nn.Parameter(initial_params['f_dc'].to(device))
        self.f_rest = torch.nn.Parameter(initial_params['f_rest'].to(device))
        self.scale_raw = torch.nn.Parameter(initial_params['scale_raw'].to(device))
        self.q_raw = torch.nn.Parameter(initial_params['q_raw'].to(device))
        
    def get_params(self):
        """Get all parameters as a dictionary."""
        return {
            'pos': self.pos,
            'opacity_raw': self.opacity_raw,
            'f_dc': self.f_dc,
            'f_rest': self.f_rest,
            'scale_raw': self.scale_raw,
            'q_raw': self.q_raw
        }
    
    def get_num_gaussians(self):
        """Get current number of Gaussians."""
        return self.pos.shape[0]
    
    def densify_and_prune(self, grads, opacity_threshold=0.01, 
                         max_grad=0.01, scale_threshold=0.01, 
                         max_screen_size=20):
        """
        Adaptive density control: split, clone, and prune Gaussians.
        
        During training, we adaptively adjust the number and placement of Gaussians:
        
        1. Split: Large Gaussians (large scale) with high gradients
           - Split into two smaller Gaussians
           - Helps refine details
        
        2. Clone: Small Gaussians with high gradients
           - Duplicate the Gaussian
           - Helps fill gaps
        
        3. Prune: Low-opacity Gaussians
           - Remove Gaussians that don't contribute
           - Keeps representation compact
        
        Args:
            grads: Dictionary of gradients for each parameter
            opacity_threshold: Minimum opacity to keep Gaussian
            max_grad: Maximum gradient magnitude for splitting/cloning
            scale_threshold: Minimum scale for splitting
            max_screen_size: Maximum screen-space size for splitting
        """
        # Compute opacity
        opacity = torch.sigmoid(self.opacity_raw)
        
        # Prune low-opacity Gaussians
        prune_mask = opacity < opacity_threshold
        self._prune_points(prune_mask)
        
        # Update grads after pruning
        if grads is not None:
            for key in grads:
                if grads[key] is not None:
                    grads[key] = grads[key][~prune_mask]
        
        # Compute gradient magnitudes
        if grads is not None and grads.get('pos') is not None:
            grad_norm = grads['pos'].norm(dim=-1)
            
            # Split large Gaussians with high gradients
            scales = torch.exp(self.scale_raw)
            max_scale = scales.max(dim=-1)[0]
            split_mask = (max_scale > scale_threshold) & (grad_norm > max_grad)
            self._split_points(split_mask)
            
            # Clone small Gaussians with high gradients
            clone_mask = (max_scale <= scale_threshold) & (grad_norm > max_grad)
            self._clone_points(clone_mask)
    
    def _prune_points(self, mask):
        """Remove Gaussians based on mask."""
        if not mask.any():
            return
        
        self.pos = torch.nn.Parameter(self.pos[~mask])
        self.opacity_raw = torch.nn.Parameter(self.opacity_raw[~mask])
        self.f_dc = torch.nn.Parameter(self.f_dc[~mask])
        self.f_rest = torch.nn.Parameter(self.f_rest[~mask])
        self.scale_raw = torch.nn.Parameter(self.scale_raw[~mask])
        self.q_raw = torch.nn.Parameter(self.q_raw[~mask])
    
    def _split_points(self, mask):
        """Split large Gaussians into two smaller ones."""
        if not mask.any():
            return
        
        N = mask.sum()
        
        # Create new positions (slightly offset)
        new_pos = self.pos[mask].clone()
        offset = torch.randn_like(new_pos) * torch.exp(self.scale_raw[mask]) * 0.1
        new_pos = new_pos + offset
        
        # Create new scales (smaller)
        new_scale_raw = self.scale_raw[mask].clone() - 0.5  # Smaller scale
        
        # Copy other parameters
        new_opacity_raw = self.opacity_raw[mask].clone()
        new_f_dc = self.f_dc[mask].clone()
        new_f_rest = self.f_rest[mask].clone()
        new_q_raw = self.q_raw[mask].clone()
        
        # Concatenate new points
        self.pos = torch.nn.Parameter(torch.cat([self.pos, new_pos], dim=0))
        self.opacity_raw = torch.nn.Parameter(torch.cat([self.opacity_raw, new_opacity_raw], dim=0))
        self.f_dc = torch.nn.Parameter(torch.cat([self.f_dc, new_f_dc], dim=0))
        self.f_rest = torch.nn.Parameter(torch.cat([self.f_rest, new_f_rest], dim=0))
        self.scale_raw = torch.nn.Parameter(torch.cat([self.scale_raw, new_scale_raw], dim=0))
        self.q_raw = torch.nn.Parameter(torch.cat([self.q_raw, new_q_raw], dim=0))
    
    def _clone_points(self, mask):
        """Clone small Gaussians."""
        if not mask.any():
            return
        
        # Simply duplicate the Gaussians
        self.pos = torch.nn.Parameter(torch.cat([self.pos, self.pos[mask]], dim=0))
        self.opacity_raw = torch.nn.Parameter(torch.cat([self.opacity_raw, self.opacity_raw[mask]], dim=0))
        self.f_dc = torch.nn.Parameter(torch.cat([self.f_dc, self.f_dc[mask]], dim=0))
        self.f_rest = torch.nn.Parameter(torch.cat([self.f_rest, self.f_rest[mask]], dim=0))
        self.scale_raw = torch.nn.Parameter(torch.cat([self.scale_raw, self.scale_raw[mask]], dim=0))
        self.q_raw = torch.nn.Parameter(torch.cat([self.q_raw, self.q_raw[mask]], dim=0))
    
    def save_checkpoint(self, path, iteration):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'pos': self.pos.cpu(),
            'opacity_raw': self.opacity_raw.cpu(),
            'f_dc': self.f_dc.cpu(),
            'f_rest': self.f_rest.cpu(),
            'scale_raw': self.scale_raw.cpu(),
            'q_raw': self.q_raw.cpu(),
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.pos = torch.nn.Parameter(checkpoint['pos'].to(self.device))
        self.opacity_raw = torch.nn.Parameter(checkpoint['opacity_raw'].to(self.device))
        self.f_dc = torch.nn.Parameter(checkpoint['f_dc'].to(self.device))
        self.f_rest = torch.nn.Parameter(checkpoint['f_rest'].to(self.device))
        self.scale_raw = torch.nn.Parameter(checkpoint['scale_raw'].to(self.device))
        self.q_raw = torch.nn.Parameter(checkpoint['q_raw'].to(self.device))
        return checkpoint.get('iteration', 0)


def train(
    data_dir,
    output_dir='output',
    iterations=30000,
    lr=0.01,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30000,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    lambda_l1=0.8,
    lambda_ssim=0.2,
    densify_until_iter=15000,
    densification_interval=100,
    opacity_reset_interval=3000,
    prune_opacity_threshold=0.01,
    max_grad=0.01,
    scale_threshold=0.01,
    checkpoint_interval=1000,
    pcd_path=None,
    num_sh_bands=3,
    device='cuda',
    num_gpus=None,
    batch_size=1,
    scale_factor=1.0,
    args=None
):
    """
    Main training function for 3D Gaussian Splatting.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save checkpoints and outputs
        iterations: Total number of training iterations
        lr: Base learning rate (for SH features)
        position_lr_init: Initial learning rate for positions
        position_lr_final: Final learning rate for positions
        position_lr_delay_mult: Learning rate delay multiplier
        position_lr_max_steps: Steps for position LR decay
        feature_lr: Learning rate for SH features
        opacity_lr: Learning rate for opacity
        scaling_lr: Learning rate for scales
        rotation_lr: Learning rate for rotations
        lambda_l1: Weight for L1 loss
        lambda_ssim: Weight for SSIM loss
        densify_until_iter: Iteration to stop densification
        densification_interval: How often to densify/prune
        opacity_reset_interval: How often to reset opacity
        prune_opacity_threshold: Opacity threshold for pruning
        max_grad: Maximum gradient for splitting/cloning
        scale_threshold: Scale threshold for splitting
        checkpoint_interval: How often to save checkpoints
        pcd_path: Path to initial point cloud (optional)
        num_sh_bands: Number of SH bands (0-3)
        device: Device to train on
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Detect number of GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    print(f"Using {num_gpus} GPU(s) for training")
    
    # Load dataset with optional scaling to reduce memory
    # Default to 0.5x scale for large images to avoid OOM
    if scale_factor == 1.0:
        # Auto-detect if we need scaling based on image size
        sample_dataset = GaussianDataset(data_dir, scale_factor=1.0)
        if len(sample_dataset) > 0:
            sample = sample_dataset[0]
            H, W = sample['H'], sample['W']
            # If images are very large (>4MP), use half resolution
            if H * W > 4_000_000:
                print(f"Large images detected ({H}x{W}), using scale_factor=0.5 to reduce memory")
                scale_factor = 0.5
                dataset = GaussianDataset(data_dir, scale_factor=scale_factor)
            else:
                dataset = sample_dataset
        else:
            dataset = sample_dataset
    else:
        if scale_factor != 1.0:
            print(f"Using scale factor {scale_factor} to reduce memory usage")
        dataset = GaussianDataset(data_dir, scale_factor=scale_factor)
    # For memory efficiency, don't force batch_size >= num_gpus
    # Process sequentially but use multiple GPUs for data loading parallelism
    effective_batch_size = batch_size
    print(f"Batch size: {effective_batch_size} (using {num_gpus} GPU(s) for data loading)")
    dataloader = DataLoader(
        dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=min(4, effective_batch_size), 
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize Gaussians
    # Try to find point cloud automatically if not provided
    if pcd_path is None:
        # Check for point cloud in data directory
        possible_pcd_paths = [
            Path(data_dir) / 'pointcloud.ply',
            Path(data_dir) / 'pointcloud.npy',
            Path(data_dir) / 'pointcloud.pt',
        ]
        for p in possible_pcd_paths:
            if p.exists():
                pcd_path = str(p)
                print(f"Found point cloud: {pcd_path}")
                break
    
    if pcd_path is not None and Path(pcd_path).exists():
        print(f"Loading point cloud from {pcd_path}...")
        points = load_point_cloud(pcd_path)
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy() if points.is_cuda else points.numpy()
        initial_params = initialize_gaussians_from_pointcloud(
            torch.from_numpy(points).float() if isinstance(points, np.ndarray) else points, 
            num_sh_bands
        )
        print(f"Initialized {len(points)} Gaussians from point cloud")
    else:
        # Initialize from random points (fallback)
        print("Warning: No point cloud provided, initializing from random points")
        # Use camera positions to estimate scene bounds
        sample = dataset[0]
        c2w = sample['c2w']
        # Get camera position (translation part of c2w)
        cam_pos = c2w[:3, 3].cpu().numpy() if isinstance(c2w, torch.Tensor) else c2w[:3, 3]
        
        # Initialize Gaussians near camera positions
        # Sample around camera position and origin
        N = 10000  # Initial number of Gaussians
        # Mix of points near origin and near camera
        points_near_origin = torch.randn(N // 2, 3) * 2.0  # [-2, 2]
        cam_pos_tensor = torch.from_numpy(cam_pos).float() if isinstance(cam_pos, np.ndarray) else cam_pos
        points_near_cam = cam_pos_tensor.unsqueeze(0) + torch.randn(N // 2, 3) * 1.0
        points = torch.cat([points_near_origin, points_near_cam], dim=0)
        
        initial_params = initialize_gaussians_from_pointcloud(points, num_sh_bands)
        print(f"Initialized {N} Gaussians from random points (around origin and camera)")
    
    # Create model on primary device
    primary_device = f'cuda:0' if num_gpus > 0 and torch.cuda.is_available() else device
    model = GaussianModel(initial_params, device=primary_device)
    device = primary_device  # Update device variable
    
    # Resume from checkpoint if provided
    start_iteration = 0
    resume_from = getattr(args, 'resume_from', None) if args is not None else None
    
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            # Try relative to output_dir
            resume_path = output_dir / resume_from
        if resume_path.exists():
            print(f"Resuming training from {resume_path}...")
            start_iteration = model.load_checkpoint(resume_path)
            print(f"Resumed from iteration {start_iteration}")
        else:
            print(f"Warning: Resume checkpoint {resume_from} not found, starting from scratch")
    
    # Setup optimizer with different learning rates for different parameters
    optimizer = optim.Adam([
        {'params': [model.pos], 'lr': position_lr_init, 'name': 'pos'},
        {'params': [model.opacity_raw], 'lr': opacity_lr, 'name': 'opacity'},
        {'params': [model.f_dc], 'lr': feature_lr, 'name': 'f_dc'},
        {'params': [model.f_rest], 'lr': feature_lr / 20.0, 'name': 'f_rest'},
        {'params': [model.scale_raw], 'lr': scaling_lr, 'name': 'scale'},
        {'params': [model.q_raw], 'lr': rotation_lr, 'name': 'rotation'},
    ], lr=lr, eps=1e-15)
    
    # Training loop
    iteration = start_iteration
    progress_bar = tqdm(range(start_iteration, iterations), desc="Training", initial=start_iteration)
    dataloader_iter = iter(dataloader)
    
    for iteration in progress_bar:
        # Get batch of views
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Handle batching - DataLoader returns batched dict with tensors
        # Check if batch is already batched (tensors have batch dimension)
        if isinstance(batch, dict) and isinstance(batch['c2w'], torch.Tensor):
            # Batched tensors - get batch size from first dimension
            if batch['c2w'].dim() == 3:  # [B, 4, 4]
                batch_size_actual = batch['c2w'].shape[0]
            elif batch['c2w'].dim() == 2:  # [4, 4] - single sample
                batch_size_actual = 1
            else:
                batch_size_actual = 1
        elif isinstance(batch, list):
            # List of dicts
            batch_size_actual = len(batch)
            # Convert to batched format
            batched = {
                'image': torch.stack([b['image'] for b in batch]),
                'c2w': torch.stack([b['c2w'] for b in batch]),
                'fx': torch.tensor([b['fx'] for b in batch]),
                'fy': torch.tensor([b['fy'] for b in batch]),
                'cx': torch.tensor([b['cx'] for b in batch]),
                'cy': torch.tensor([b['cy'] for b in batch]),
                'H': [b['H'] for b in batch],
                'W': [b['W'] for b in batch],
            }
            batch = batched
        else:
            # Single sample
            batch_size_actual = 1
        
        # Update position learning rate (exponential decay)
        if iteration < position_lr_max_steps:
            position_lr = position_lr_init * (
                position_lr_final / position_lr_init
            ) ** (iteration / position_lr_max_steps)
        else:
            position_lr = position_lr_final
        
        # Apply position LR delay
        if iteration < position_lr_delay_mult * position_lr_max_steps:
            position_lr *= 0.01
        
        optimizer.param_groups[0]['lr'] = position_lr
        
        # Forward pass - process batch
        optimizer.zero_grad()
        
        # Build covariance (shared across batch)
        sigma = build_sigma_from_params(model.scale_raw, model.q_raw)
        
        total_loss = 0
        total_loss_dict = {'l1': 0, 'ssim': 0}
        
        # Process each view in batch sequentially
        # Keep everything on primary device to avoid device mismatches
        # Memory is managed by clearing cache and using smaller images
        for i in range(batch_size_actual):
            # Extract single view from batched tensors
            if isinstance(batch['image'], torch.Tensor) and batch['image'].dim() > 3:
                # Batched tensor [B, H, W, 3]
                image_gt = batch['image'][i].to(device)
            else:
                image_gt = batch['image'][i]
                if not isinstance(image_gt, torch.Tensor):
                    image_gt = torch.tensor(image_gt)
                image_gt = image_gt.to(device)
            
            if isinstance(batch['c2w'], torch.Tensor) and batch['c2w'].dim() == 3:
                # Batched tensor [B, 4, 4]
                c2w = batch['c2w'][i].to(device)
            else:
                c2w = batch['c2w'][i]
                if not isinstance(c2w, torch.Tensor):
                    c2w = torch.tensor(c2w, dtype=torch.float32)
                c2w = c2w.to(device)
            
            # Extract scalar values
            if isinstance(batch['fx'], torch.Tensor):
                fx, fy = batch['fx'][i].item(), batch['fy'][i].item()
                cx, cy = batch['cx'][i].item(), batch['cy'][i].item()
            else:
                fx, fy = batch['fx'][i], batch['fy'][i]
                cx, cy = batch['cx'][i], batch['cy'][i]
            
            H, W = batch['H'][i], batch['W'][i]
            
            # Evaluate colors (model stays on primary device)
            colors = evaluate_sh(model.f_dc, model.f_rest, model.pos, c2w)
            
            # Render (enable gradients for training)
            rendered = render(
                model.pos, colors, model.opacity_raw, sigma, c2w,
                H, W, fx, fy, cx, cy
            )
            
            # Compute loss
            loss, loss_dict = compute_loss(rendered, image_gt, lambda_l1, lambda_ssim)
            
            # Accumulate loss (divide by batch size for averaging)
            if i == 0:
                total_loss = loss / batch_size_actual
                total_loss_dict['l1'] = loss_dict['l1'] / batch_size_actual
                total_loss_dict['ssim'] = loss_dict['ssim'] / batch_size_actual
            else:
                total_loss = total_loss + loss / batch_size_actual
                total_loss_dict['l1'] += loss_dict['l1'] / batch_size_actual
                total_loss_dict['ssim'] += loss_dict['ssim'] / batch_size_actual
            
            # Clear intermediate tensors to free memory immediately
            del rendered, colors, image_gt, c2w, loss
            # Clear cache periodically to prevent OOM
            if (i + 1) % 2 == 0:  # Clear every 2 views
                torch.cuda.empty_cache()
        
        # Backward pass
        total_loss.backward()
        
        # Clear cache before optimizer step
        torch.cuda.empty_cache()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.pos, max_norm=1.0)
        
        optimizer.step()
        
        # Clear cache after optimizer step
        torch.cuda.empty_cache()
        
        # Densification and pruning
        if iteration < densify_until_iter and iteration % densification_interval == 0:
            # Get gradients
            grads = {
                'pos': model.pos.grad,
                'opacity_raw': model.opacity_raw.grad,
            }
            
            # Densify and prune
            model.densify_and_prune(
                grads,
                opacity_threshold=prune_opacity_threshold,
                max_grad=max_grad,
                scale_threshold=scale_threshold
            )
            
            # Recreate optimizer with new parameters
            optimizer = optim.Adam([
                {'params': [model.pos], 'lr': position_lr, 'name': 'pos'},
                {'params': [model.opacity_raw], 'lr': opacity_lr, 'name': 'opacity'},
                {'params': [model.f_dc], 'lr': feature_lr, 'name': 'f_dc'},
                {'params': [model.f_rest], 'lr': feature_lr / 20.0, 'name': 'f_rest'},
                {'params': [model.scale_raw], 'lr': scaling_lr, 'name': 'scale'},
                {'params': [model.q_raw], 'lr': rotation_lr, 'name': 'rotation'},
            ], lr=lr, eps=1e-15)
        
        # Opacity reset (helps with optimization)
        if iteration % opacity_reset_interval == 0:
            opacity = torch.sigmoid(model.opacity_raw)
            mask = opacity < 0.01
            if mask.any():
                model.opacity_raw.data[mask] = torch.logit(torch.clamp(opacity[mask] + 0.01, 0, 1))
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'l1': f"{total_loss_dict['l1']:.4f}",
            'ssim': f"{total_loss_dict['ssim']:.4f}",
            'gaussians': model.get_num_gaussians(),
            'lr_pos': f"{position_lr:.6f}",
            'batch': batch_size_actual
        })
        
        # Save checkpoint
        if iteration % checkpoint_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_{iteration:06d}.pt'
            model.save_checkpoint(checkpoint_path, iteration)
            
            # Also save individual parameter files (for inference)
            torch.save(model.pos.cpu(), output_dir / f'pos_{iteration}.pt')
            torch.save(model.opacity_raw.cpu(), output_dir / f'opacity_raw_{iteration}.pt')
            torch.save(model.f_dc.cpu(), output_dir / f'f_dc_{iteration}.pt')
            torch.save(model.f_rest.cpu(), output_dir / f'f_rest_{iteration}.pt')
            torch.save(model.scale_raw.cpu(), output_dir / f'scale_raw_{iteration}.pt')
            torch.save(model.q_raw.cpu(), output_dir / f'q_rot_{iteration}.pt')
    
    # Final save
    final_checkpoint = output_dir / 'checkpoint_final.pt'
    model.save_checkpoint(final_checkpoint, iteration)
    
    print(f"\nTraining complete! Final checkpoint saved to {final_checkpoint}")
    print(f"Final number of Gaussians: {model.get_num_gaussians()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for checkpoints')
    parser.add_argument('--pcd_path', type=str, default=None,
                        help='Path to initial point cloud')
    parser.add_argument('--iterations', type=int, default=30000,
                        help='Number of training iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (number of views per iteration)')
    parser.add_argument('--scale_factor', type=float, default=0.5,
                        help='Scale factor for images (1.0 = full res, 0.5 = half res) to reduce memory. Default 0.5 for large images.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint (path to checkpoint file)')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        pcd_path=args.pcd_path,
        device=args.device,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        args=args
    )

