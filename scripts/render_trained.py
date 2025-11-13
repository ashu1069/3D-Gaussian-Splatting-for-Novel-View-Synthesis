"""
Simple script to render images from trained 3D Gaussians.

Usage:
    python render_trained.py --checkpoint_dir output/garden --iteration 6000 --output_dir renders/garden
    python render_trained.py --checkpoint_dir output/garden --iteration final --output_dir renders/garden
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import subprocess
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaussian_splatting.gaussian import build_sigma_from_params
from gaussian_splatting.spherical_harmonics import evaluate_sh
from gaussian_splatting.render import render
from gaussian_splatting.data_loader import GaussianDataset


def create_orbit_trajectory(center, radius=3.0, num_frames=60, elevation=0.0):
    """
    Create a circular orbit camera trajectory around a center point.
    
    Args:
        center: Center point [x, y, z]
        radius: Orbit radius
        num_frames: Number of frames in orbit
        elevation: Elevation angle (radians)
        
    Returns:
        List of c2w matrices [num_frames, 4, 4]
    """
    c2ws = []
    center = np.array(center)
    
    for i in range(num_frames):
        # Circular orbit
        angle = 2 * np.pi * i / num_frames
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = radius * np.sin(elevation)
        
        # Camera position
        cam_pos = center + np.array([x, y, z])
        
        # Look at center
        forward = center - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # Right vector (perpendicular to forward and up)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Up vector
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Build c2w matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up  # Negative because camera looks down -Y
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        
        c2ws.append(c2w)
    
    return np.array(c2ws)


def render_trained_model(
    checkpoint_dir,
    data_dir,
    iteration='final',
    output_dir='renders',
    render_training_views=False,
    render_orbit=True,
    orbit_frames=60,
    scale_factor=1.0,
    benchmark_only=False,
    video_fps=30,
    no_video=False
):
    """
    Render images from a trained 3D Gaussian Splatting model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        data_dir: Directory containing training data (for camera params)
        iteration: Which iteration to render ('final' or number like 6000)
        output_dir: Where to save rendered images
        render_training_views: If True, render all training views
        render_orbit: If True, render a circular orbit
        orbit_frames: Number of frames in orbit
        scale_factor: Image scale factor (1.0 = full res, 0.5 = half res)
    """
    checkpoint_dir = Path(checkpoint_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if checkpoint directory exists
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}\n"
            f"Available directories: {list(checkpoint_dir.parent.glob('*')) if checkpoint_dir.parent.exists() else 'N/A'}"
        )
    
    # Determine checkpoint path
    if iteration == 'final':
        checkpoint_path = checkpoint_dir / 'checkpoint_final.pt'
        iteration_suffix = 'final'
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_{int(iteration):06d}.pt'
        iteration_suffix = str(int(iteration))
    
    # Check if we should load from checkpoint file or individual files
    load_from_checkpoint = False
    if checkpoint_path.exists():
        load_from_checkpoint = True
    elif not checkpoint_path.exists():
        # Try loading individual parameter files
        print(f"Checkpoint {checkpoint_path} not found, trying individual files...")
        pos_path = checkpoint_dir / f'pos_{iteration_suffix}.pt'
        opacity_path = checkpoint_dir / f'opacity_raw_{iteration_suffix}.pt'
        f_dc_path = checkpoint_dir / f'f_dc_{iteration_suffix}.pt'
        f_rest_path = checkpoint_dir / f'f_rest_{iteration_suffix}.pt'
        scale_path = checkpoint_dir / f'scale_raw_{iteration_suffix}.pt'
        q_path = checkpoint_dir / f'q_rot_{iteration_suffix}.pt'
        
        if not all(p.exists() for p in [pos_path, opacity_path, f_dc_path, f_rest_path, scale_path, q_path]):
            # Try to find any available checkpoint
            available_checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
            if len(available_checkpoints) > 0:
                latest_checkpoint = available_checkpoints[-1]
                print(f"Warning: Checkpoint for iteration '{iteration}' not found.")
                print(f"Available checkpoints: {[c.stem for c in available_checkpoints]}")
                print(f"Using latest checkpoint: {latest_checkpoint.name}")
                checkpoint_path = latest_checkpoint
                load_from_checkpoint = True
            else:
                # List what files actually exist
                existing_files = list(checkpoint_dir.glob('*.pt'))
                raise FileNotFoundError(
                    f"Could not find checkpoint files for iteration {iteration}.\n"
                    f"Checked directory: {checkpoint_dir}\n"
                    f"Looking for: checkpoint_{iteration_suffix if iteration != 'final' else 'final'}.pt\n"
                    f"Or individual files: pos_{iteration_suffix}.pt, opacity_raw_{iteration_suffix}.pt, etc.\n"
                    f"Files found in directory: {[f.name for f in existing_files] if existing_files else 'None'}\n"
                    f"Hint: Make sure training has completed and checkpoints were saved."
                )
        else:
            # Individual files exist, use them
            load_from_checkpoint = False
    
    # Load parameters
    if load_from_checkpoint:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        pos = checkpoint['pos'].cuda()
        opacity_raw = checkpoint['opacity_raw'].cuda()
        f_dc = checkpoint['f_dc'].cuda()
        f_rest = checkpoint['f_rest'].cuda()
        scale_raw = checkpoint['scale_raw'].cuda()
        q_raw = checkpoint['q_raw'].cuda()
    else:
        print("Loading from individual parameter files...")
        pos = torch.load(pos_path).cuda()
        opacity_raw = torch.load(opacity_path).cuda()
        f_dc = torch.load(f_dc_path).cuda()
        f_rest = torch.load(f_rest_path).cuda()
        scale_raw = torch.load(scale_path).cuda()
        q_raw = torch.load(q_path).cuda()
    
    print(f"Loaded {len(pos)} Gaussians")
    
    # Load camera parameters
    cam_meta_path = data_dir / 'cam_meta.npy'
    if not cam_meta_path.exists():
        raise FileNotFoundError(f"Camera metadata not found at {cam_meta_path}")
    
    cam_params = np.load(cam_meta_path, allow_pickle=True).item()
    
    # Build covariance matrices
    sigma = build_sigma_from_params(scale_raw, q_raw)
    
    # Scale intrinsics
    H_src = cam_params['height']
    W_src = cam_params['width']
    H = int(H_src * scale_factor)
    W = int(W_src * scale_factor)
    fx = cam_params['fx'] * scale_factor
    fy = cam_params['fy'] * scale_factor
    cx = cam_params.get('cx', W_src / 2) * scale_factor
    cy = cam_params.get('cy', H_src / 2) * scale_factor
    
    print(f"Rendering at resolution {W}x{H} (scale_factor={scale_factor})")
    
    # Render training views if requested
    if render_training_views:
        print("Rendering training views...")
        dataset = GaussianDataset(data_dir, scale_factor=scale_factor)
        training_output_dir = output_dir / 'training_views'
        training_output_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                sample = dataset[idx]
                c2w = sample['c2w'].cuda()
                
                # Evaluate colors
                colors = evaluate_sh(f_dc, f_rest, pos, c2w)
                
                # Render with optimized parameters
                img = render(pos, colors, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy,
                           pix_guard=32, chi_square_clip=6.25, alpha_cutoff=1/128.)
                
                # Save
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(training_output_dir / f'train_{idx:04d}.png')
        
        print(f"Saved {len(dataset)} training views to {training_output_dir}")
    
    # Render orbit if requested
    if render_orbit:
        print("Rendering orbit trajectory...")
        orbit_output_dir = output_dir / 'orbit'
        orbit_output_dir.mkdir(exist_ok=True)
        
        # Estimate scene center and orbit radius
        # Prefer using training camera poses for better orbit estimation
        pos_np = pos.detach().cpu().numpy()
        valid_mask = np.isfinite(pos_np).all(axis=1)
        
        # Try to use training camera poses first (more reliable)
        use_camera_poses = True
        try:
            dataset = GaussianDataset(data_dir, scale_factor=1.0)
            if len(dataset) > 0:
                # Extract camera positions and look-at directions from training poses
                cam_positions = []
                cam_forwards = []
                for idx in range(min(len(dataset), 100)):  # Sample up to 100 cameras
                    sample = dataset[idx]
                    c2w = sample['c2w']
                    if isinstance(c2w, torch.Tensor):
                        c2w = c2w.numpy()
                    # Camera position is in c2w[:3, 3]
                    cam_pos = c2w[:3, 3]
                    cam_forward = c2w[:3, 2]  # Camera forward direction
                    if np.isfinite(cam_pos).all() and np.isfinite(cam_forward).all():
                        cam_positions.append(cam_pos)
                        cam_forwards.append(cam_forward)
                
                if len(cam_positions) > 0:
                    cam_positions = np.array(cam_positions)
                    cam_forwards = np.array(cam_forwards)
                    
                    # Estimate scene center as average of camera look-at points
                    # Each camera looks along its forward direction
                    # Estimate look-at point as camera_pos + forward * typical_distance
                    typical_dist = 5.0  # Typical distance cameras look at
                    look_at_points = cam_positions + cam_forwards * typical_dist
                    scene_center = look_at_points.mean(axis=0)
                    
                    # Estimate radius from camera spread (how far cameras are from center)
                    distances = np.linalg.norm(cam_positions - scene_center, axis=1)
                    cam_spread = distances.max()
                    # Orbit radius should be similar to camera spread, maybe slightly larger
                    orbit_radius = max(cam_spread * 1.2, 3.0)  # At least 3 units
                    orbit_radius = min(orbit_radius, 20.0)  # Cap at 20 units
                    
                    print(f"Scene center (from training cameras): {scene_center}")
                    print(f"Camera spread: {cam_spread:.2f}, Orbit radius: {orbit_radius:.2f}")
                    use_camera_poses = True
                else:
                    use_camera_poses = False
            else:
                use_camera_poses = False
        except Exception as e:
            print(f"Warning: Could not use camera poses: {e}")
            use_camera_poses = False
        
        # Fallback to Gaussian positions if camera poses didn't work
        if not use_camera_poses:
            if valid_mask.sum() == 0:
                # Ultimate fallback: use origin
                scene_center = np.array([0.0, 0.0, 0.0])
                orbit_radius = 5.0
                print(f"Using default scene center: {scene_center}, radius: {orbit_radius}")
            else:
                # Use valid Gaussian positions
                pos_valid = pos_np[valid_mask]
                scene_center = pos_valid.mean(axis=0)
                print(f"Scene center (from Gaussians): {scene_center}")
                
                # Estimate orbit radius (distance from center to furthest Gaussian)
                distances = np.linalg.norm(pos_valid - scene_center, axis=1)
                max_dist = distances.max()
                # Use a more reasonable orbit radius
                orbit_radius = min(max_dist * 1.2, 20.0)  # Cap at 20 units max
                print(f"Scene extent: {max_dist:.2f}, Orbit radius: {orbit_radius:.2f}")
        
        # Create orbit trajectory
        orbit_c2ws = create_orbit_trajectory(
            center=scene_center,
            radius=orbit_radius,
            num_frames=orbit_frames,
            elevation=0.0
        )
        
        # Measure FPS for performance evaluation
        import time
        render_times = []
        
        with torch.no_grad():
            # Warmup run (first frame is often slower due to CUDA initialization)
            if len(orbit_c2ws) > 0:
                c2w_warmup = torch.from_numpy(orbit_c2ws[0]).float().cuda()
                colors_warmup = evaluate_sh(f_dc, f_rest, pos, c2w_warmup)
                _ = render(pos, colors_warmup, opacity_raw, sigma, c2w_warmup, H, W, fx, fy, cx, cy,
                          pix_guard=32, chi_square_clip=6.25, alpha_cutoff=1/128.)
                torch.cuda.synchronize()  # Wait for GPU to finish
            
            # Actual rendering with timing
            for i, c2w_np in enumerate(tqdm(orbit_c2ws, desc="Rendering")):
                c2w = torch.from_numpy(c2w_np).float().cuda()
                
                # Start timing
                torch.cuda.synchronize()  # Ensure previous operations are done
                start_time = time.time()
                
                # Evaluate colors
                colors = evaluate_sh(f_dc, f_rest, pos, c2w)
                
                # Render with optimized parameters
                img = render(pos, colors, opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy,
                           pix_guard=32, chi_square_clip=6.25, alpha_cutoff=1/128.)
                
                # Wait for GPU to finish
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Record render time (excluding I/O)
                render_time = end_time - start_time
                render_times.append(render_time)
                
                # Save (skip if benchmark only)
                if not benchmark_only:
                    img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(orbit_output_dir / f'frame_{i:04d}.png')
        
        # Calculate FPS statistics
        render_times = np.array(render_times)
        fps_values = 1.0 / render_times  # FPS = 1 / render_time_per_frame
        
        print(f"\n{'='*60}")
        print(f"RENDERING PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Resolution: {W}x{H} (scale_factor={scale_factor})")
        print(f"Number of Gaussians: {len(pos):,}")
        print(f"Number of frames rendered: {orbit_frames}")
        print(f"\nRender Time per Frame:")
        print(f"  Mean:   {render_times.mean()*1000:.2f} ms")
        print(f"  Median: {np.median(render_times)*1000:.2f} ms")
        print(f"  Min:    {render_times.min()*1000:.2f} ms")
        print(f"  Max:    {render_times.max()*1000:.2f} ms")
        print(f"  Std:    {render_times.std()*1000:.2f} ms")
        print(f"\nFPS (Frames Per Second):")
        print(f"  Mean:   {fps_values.mean():.2f} FPS")
        print(f"  Median: {np.median(fps_values):.2f} FPS")
        print(f"  Min:    {fps_values.min():.2f} FPS")
        print(f"  Max:    {fps_values.max():.2f} FPS")
        print(f"{'='*60}")
        print(f"Saved {orbit_frames} orbit frames to {orbit_output_dir}")
        
        # Create video from rendered frames
        if not benchmark_only and not no_video:
            video_path = output_dir / 'orbit_video.mp4'
            print(f"\nCreating video from rendered frames...")
            create_video_from_frames(orbit_output_dir, video_path, fps=video_fps)
            if video_path.exists():
                print(f"✓ Video saved to: {video_path}")
    
    print(f"\nRendering complete! Images saved to {output_dir}")


def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create a video from rendered frames.
    
    First tries to use ffmpeg command-line tool (if available).
    Falls back to imageio (Python library) if ffmpeg is not available.
    
    Args:
        frames_dir: Directory containing frame images (frame_0000.png, frame_0001.png, ...)
        output_path: Path to save the output video (.mp4)
        fps: Frames per second for the video
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    # Get list of frame files
    frame_files = sorted(frames_dir.glob('frame_*.png'))
    if len(frame_files) == 0:
        print("Warning: No frame files found. Skipping video creation.")
        return
    
    # Try method 1: Use ffmpeg command-line tool (fastest, best quality)
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        
        # Use ffmpeg to create video
        input_pattern = str(frames_dir / 'frame_%04d.png')
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',  # Compatibility with most players
            '-crf', '18',  # High quality (lower = better quality, 18-23 is good)
            '-preset', 'medium',  # Encoding speed vs compression
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Video created using ffmpeg")
        return
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Method 2: Use imageio (Python library, works without system ffmpeg)
        try:
            import imageio.v2 as imageio
            print("Using imageio to create video (ffmpeg not found)...")
            
            # Read all frames
            frames = []
            for frame_file in tqdm(frame_files, desc="Loading frames"):
                img = imageio.imread(frame_file)
                frames.append(img)
            
            # Write video
            imageio.mimwrite(
                str(output_path),
                frames,
                fps=fps,
                codec='libx264',
                quality=8,  # 0-10, higher = better quality
                pixelformat='yuv420p'  # Compatibility
            )
            print(f"Video created using imageio")
            return
            
        except ImportError:
            print("Warning: Neither ffmpeg nor imageio found.")
            print("Install one of the following:")
            print("  - System ffmpeg: sudo apt-get install ffmpeg")
            print("  - Python imageio: pip install imageio imageio-ffmpeg")
            return
        except Exception as e:
            print(f"Error creating video with imageio: {e}")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render images from trained 3D Gaussians')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--iteration', type=str, default='final',
                        help='Which iteration to render (e.g., 6000 or "final")')
    parser.add_argument('--output_dir', type=str, default='renders',
                        help='Output directory for rendered images')
    parser.add_argument('--render_training', action='store_true',
                        help='Render all training views')
    parser.add_argument('--render_orbit', action='store_true', default=True,
                        help='Render circular orbit (default: True)')
    parser.add_argument('--orbit_frames', type=int, default=60,
                        help='Number of frames in orbit')
    parser.add_argument('--scale_factor', type=float, default=1.0,
                        help='Image scale factor (1.0 = full res)')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='Only run benchmark (skip saving images)')
    parser.add_argument('--video_fps', type=int, default=30,
                        help='FPS for output video (default: 30)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video creation')
    
    args = parser.parse_args()
    
    render_trained_model(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        iteration=args.iteration,
        output_dir=args.output_dir,
        render_training_views=args.render_training,
        render_orbit=args.render_orbit,
        orbit_frames=args.orbit_frames,
        scale_factor=args.scale_factor,
        benchmark_only=args.benchmark_only,
        video_fps=args.video_fps,
        no_video=args.no_video
    )

