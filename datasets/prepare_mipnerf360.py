"""
Prepare Mip-NeRF 360 dataset for training.

This script downloads and converts Mip-NeRF 360 dataset format to the format
expected by our training pipeline.

Mip-NeRF 360 dataset structure:
- data/ (root directory)
  - images/ (or images_4, images_8 for downsampled)
  - transforms_train.json
  - transforms_val.json
  - transforms_test.json
  - sparse/0/ (COLMAP output, optional)
    - points3D.bin (point cloud)

Our training format expects:
- images/ (training images)
- cam_meta.npy (camera metadata)
- poses.npy (camera poses, optional)
- pointcloud.ply (initial point cloud, optional)
"""

import json
import numpy as np
import torch
from pathlib import Path
import os
import shutil
from PIL import Image
import struct
from tqdm import tqdm


def read_binary_point_cloud(colmap_points_path):
    """
    Read COLMAP binary point cloud file.
    
    COLMAP binary format for points3D.bin:
    - uint64: point3D_id
    - double[3]: xyz
    - double: error
    - uint8: track_length
    - uint64[track_length]: image_ids
    - uint32[track_length]: point2D_idxs
    
    Note: This is a variable-length format, so we need to read track_length first.
    
    Args:
        colmap_points_path: Path to points3D.bin file
        
    Returns:
        Points array [N, 3] and colors [N, 3]
    """
    points = []
    colors = []
    
    try:
        with open(colmap_points_path, 'rb') as fid:
            # Read number of points (first 8 bytes)
            num_points_data = fid.read(8)
            if len(num_points_data) < 8:
                raise ValueError("File too short or empty")
            
            num_points = struct.unpack('<Q', num_points_data)[0]
            
            for _ in range(num_points):
                # Read point ID (uint64, 8 bytes)
                point_id_data = fid.read(8)
                if len(point_id_data) < 8:
                    break
                point_id = struct.unpack('<Q', point_id_data)[0]
                
                # Read xyz (3 doubles, 24 bytes)
                xyz_data = fid.read(24)
                if len(xyz_data) < 24:
                    break
                xyz = struct.unpack('<ddd', xyz_data)
                
                # Read error (double, 8 bytes)
                error_data = fid.read(8)
                if len(error_data) < 8:
                    break
                error = struct.unpack('<d', error_data)[0]
                
                # Read track_length (uint8, 1 byte)
                track_length_data = fid.read(1)
                if len(track_length_data) < 1:
                    break
                track_length = struct.unpack('<B', track_length_data)[0]
                
                # Skip image_ids (uint64[track_length], 8*track_length bytes)
                fid.read(8 * track_length)
                
                # Read point2D_idxs to get colors (uint32[track_length], 4*track_length bytes)
                # Actually, COLMAP doesn't store colors in points3D.bin
                # Colors are typically computed from images or stored separately
                fid.read(4 * track_length)
                
                points.append(xyz)
                # Use a default color (white/gray) since colors aren't in the binary file
                colors.append([128, 128, 128])  # Gray color
                
    except Exception as e:
        print(f"Error reading COLMAP binary file: {e}")
        print("Trying alternative format...")
        # Try reading as if it's a simpler format
        try:
            with open(colmap_points_path, 'rb') as fid:
                num_points = struct.unpack('<Q', fid.read(8))[0]
                for _ in range(min(num_points, 1000000)):  # Limit to prevent memory issues
                    try:
                        # Try reading fixed-size record
                        data = fid.read(61)  # Some COLMAP versions use fixed size
                        if len(data) < 61:
                            break
                        # Parse: id(8) + xyz(24) + error(8) + track(1) + rest(20)
                        point_id = struct.unpack('<Q', data[0:8])[0]
                        xyz = struct.unpack('<ddd', data[8:32])
                        points.append(xyz)
                        colors.append([128, 128, 128])
                    except:
                        break
        except Exception as e2:
            raise ValueError(f"Could not read COLMAP point cloud: {e2}")
    
    if len(points) == 0:
        raise ValueError("No points found in COLMAP file")
    
    return np.array(points), np.array(colors)


def load_transforms_json(json_path):
    """
    Load camera transforms from Mip-NeRF 360 JSON file.
    
    Args:
        json_path: Path to transforms_train.json or similar
        
    Returns:
        Dictionary with camera parameters
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def load_poses_bounds(poses_bounds_path):
    """
    Load camera poses and bounds from Mip-NeRF 360 poses_bounds.npy file.
    
    The poses_bounds.npy file contains [N, 17] array where:
    - First 12 values: 3x4 camera-to-world matrix (row-major)
    - Next 2 values: near and far bounds
    - Last 3 values: image dimensions (height, width, focal length)
    
    Args:
        poses_bounds_path: Path to poses_bounds.npy
        
    Returns:
        Dictionary with camera parameters
    """
    poses_bounds = np.load(poses_bounds_path)
    N = poses_bounds.shape[0]
    
    # Extract poses (3x4 matrices)
    poses = poses_bounds[:, :12].reshape(N, 3, 4)
    
    # Convert to 4x4 homogeneous matrices
    c2w_matrices = np.zeros((N, 4, 4), dtype=np.float32)
    c2w_matrices[:, :3, :4] = poses
    c2w_matrices[:, 3, 3] = 1.0
    
    # Extract bounds
    bounds = poses_bounds[:, 12:14]  # [N, 2] - near and far
    
    # Extract image dimensions and focal length
    # The format varies, but typically: [height, width, focal] or [focal, height, width]
    # We'll use the last 3 values
    img_params = poses_bounds[:, 14:17]  # [N, 3]
    
    # Get image dimensions from first image
    # Try to infer from the values - typically focal is largest
    # Or we can read from actual images
    return {
        'c2w': c2w_matrices,
        'bounds': bounds,
        'img_params': img_params,
        'num_images': N
    }


def convert_transforms_to_cameras(transforms_data, data_dir):
    """
    Convert Mip-NeRF 360 transforms to our camera format.
    
    Args:
        transforms_data: Loaded JSON data
        data_dir: Root data directory
        
    Returns:
        Dictionary with:
        - c2w_matrices: List of camera-to-world matrices [N, 4, 4]
        - intrinsics: Dictionary with fx, fy, cx, cy, height, width
        - image_paths: List of image paths
    """
    frames = transforms_data['frames']
    
    # Get image dimensions from first image
    first_image_path = Path(data_dir) / frames[0]['file_path']
    if not first_image_path.exists():
        # Try with different path formats
        possible_paths = [
            first_image_path,
            Path(data_dir) / 'images' / first_image_path.name,
            Path(data_dir) / first_image_path.name,
        ]
        for p in possible_paths:
            if p.exists():
                first_image_path = p
                break
    
    img = Image.open(first_image_path)
    height, width = img.height, img.width
    
    # Camera intrinsics
    camera_angle_x = transforms_data.get('camera_angle_x', None)
    if camera_angle_x is None:
        # Estimate from image dimensions (default FOV)
        camera_angle_x = np.pi / 4  # 45 degrees
    
    fx = width / (2 * np.tan(camera_angle_x / 2))
    fy = fx  # Assume square pixels
    
    intrinsics = {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(width / 2),
        'cy': float(height / 2),
        'height': int(height),
        'width': int(width)
    }
    
    # Extract camera poses
    c2w_matrices = []
    image_paths = []
    
    for frame in frames:
        # Get transform matrix
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        c2w_matrices.append(transform_matrix)
        
        # Get image path
        file_path = frame['file_path']
        # Remove leading './' if present
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        image_path = Path(data_dir) / file_path
        if not image_path.exists():
            # Try images subdirectory
            image_path = Path(data_dir) / 'images' / Path(file_path).name
        
        image_paths.append(str(image_path))
    
    return {
        'c2w_matrices': np.array(c2w_matrices),
        'intrinsics': intrinsics,
        'image_paths': image_paths
    }


def prepare_mipnerf360_dataset(
    input_dir,
    output_dir,
    scene_name='garden',
    use_colmap_points=True,
    image_downsample=1
):
    """
    Prepare Mip-NeRF 360 dataset for training.
    
    Args:
        input_dir: Directory containing Mip-NeRF 360 dataset
        output_dir: Output directory for prepared dataset
        scene_name: Name of the scene (for organization)
        use_colmap_points: Whether to use COLMAP point cloud if available
        image_downsample: Downsample factor for images (1 = original, 2 = half size)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check for poses_bounds.npy (Mip-NeRF 360 format) or transforms_train.json (NeRF format)
    poses_bounds_path = input_dir / 'poses_bounds.npy'
    transforms_train_path = input_dir / 'transforms_train.json'
    
    if poses_bounds_path.exists():
        # Mip-NeRF 360 format (poses_bounds.npy)
        print(f"Loading poses from {poses_bounds_path}")
        poses_data = load_poses_bounds(poses_bounds_path)
        
        # Get image directory
        images_dir = input_dir / 'images'
        if not images_dir.exists():
            # Try images_2, images_4, images_8
            for img_dir_name in ['images_2', 'images_4', 'images_8']:
                test_dir = input_dir / img_dir_name
                if test_dir.exists():
                    images_dir = test_dir
                    break
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Could not find images directory in {input_dir}")
        
        # Get image files
        image_files = sorted(list(images_dir.glob('*.jpg')) + 
                            list(images_dir.glob('*.png')) +
                            list(images_dir.glob('*.JPG')) +
                            list(images_dir.glob('*.PNG')))
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")
        
        # Get image dimensions from first image
        first_img = Image.open(image_files[0])
        height, width = first_img.height, first_img.width
        
        # Extract focal length from poses_bounds
        # Format: [N, 17] where last 3 values are [height, width, focal]
        img_params = poses_data['img_params']  # [N, 3]
        
        # The format is typically [height, width, focal] in pixels
        # Use the first row to get the focal length (should be consistent)
        focal_pixels = img_params[0, 2]  # Third value is focal length
        
        # Verify focal is reasonable (should be similar to image width for typical cameras)
        if focal_pixels < 100 or focal_pixels > 20000:
            # Fallback: estimate from image dimensions (assume ~50 degree FOV)
            focal_pixels = width / (2 * np.tan(np.pi / 6))  # ~60 degree FOV
            print(f"Warning: Using estimated focal length: {focal_pixels:.1f}")
        else:
            print(f"Using focal length from poses_bounds: {focal_pixels:.1f}")
        
        # Create camera data structure
        camera_data = {
            'c2w_matrices': poses_data['c2w'],
            'intrinsics': {
                'fx': float(focal_pixels),
                'fy': float(focal_pixels),  # Assume square pixels
                'cx': float(width / 2),
                'cy': float(height / 2),
                'height': int(height),
                'width': int(width)
            },
            'image_paths': [str(f) for f in image_files]
        }
        
        print(f"Found {len(image_files)} images")
        print(f"Image dimensions: {width}x{height}")
        print(f"Focal length: {focal_pixels:.1f} pixels")
        
    elif transforms_train_path.exists():
        # NeRF format (transforms_train.json)
        print(f"Loading transforms from {transforms_train_path}")
        transforms_data = load_transforms_json(transforms_train_path)
    
        # Convert to our format
        print("Converting camera parameters...")
        camera_data = convert_transforms_to_cameras(transforms_data, input_dir)
    else:
        raise FileNotFoundError(
            f"Could not find poses_bounds.npy or transforms_train.json in {input_dir}. "
            "Please ensure the dataset is properly downloaded."
        )
    
    # Create images directory
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Copy and optionally downsample images
    print(f"Processing {len(camera_data['image_paths'])} images...")
    for i, img_path in enumerate(tqdm(camera_data['image_paths'])):
        src_path = Path(img_path)
        if not src_path.exists():
            print(f"Warning: Image not found: {src_path}")
            continue
        
        dst_path = images_dir / f"{i:06d}.jpg"
        
        if image_downsample > 1:
            # Downsample image
            img = Image.open(src_path)
            new_size = (img.width // image_downsample, img.height // image_downsample)
            img = img.resize(new_size, Image.LANCZOS)
            img.save(dst_path, 'JPEG', quality=95)
        else:
            # Copy as-is
            shutil.copy2(src_path, dst_path)
    
    # Update intrinsics if downsampled
    if image_downsample > 1:
        camera_data['intrinsics']['fx'] /= image_downsample
        camera_data['intrinsics']['fy'] /= image_downsample
        camera_data['intrinsics']['cx'] /= image_downsample
        camera_data['intrinsics']['cy'] /= image_downsample
        camera_data['intrinsics']['height'] //= image_downsample
        camera_data['intrinsics']['width'] //= image_downsample
    
    # Save camera metadata
    cam_meta_path = output_dir / 'cam_meta.npy'
    np.save(cam_meta_path, camera_data['intrinsics'])
    print(f"Saved camera metadata to {cam_meta_path}")
    
    # Save camera poses
    poses_path = output_dir / 'poses.npy'
    np.save(poses_path, camera_data['c2w_matrices'])
    print(f"Saved {len(camera_data['c2w_matrices'])} camera poses to {poses_path}")
    
    # Try to load point cloud from COLMAP
    if use_colmap_points:
        colmap_points_path = input_dir / 'sparse' / '0' / 'points3D.bin'
        if colmap_points_path.exists():
            print("Loading COLMAP point cloud...")
            try:
                points, colors = read_binary_point_cloud(colmap_points_path)
                
                # Save as PLY
                ply_path = output_dir / 'pointcloud.ply'
                save_ply(points, colors, ply_path)
                print(f"Saved point cloud ({len(points)} points) to {ply_path}")
            except Exception as e:
                print(f"Warning: Could not load COLMAP point cloud: {e}")
                print("You can still train without a point cloud (will use random initialization)")
        else:
            print("No COLMAP point cloud found. Training will use random initialization.")
    
    print(f"\nDataset preparation complete!")
    print(f"Prepared dataset location: {output_dir}")
    print(f"\nTo train, run:")
    print(f"  python train.py --data_dir {output_dir} --output_dir output/{scene_name}")


def save_ply(points, colors, output_path):
    """
    Save point cloud as PLY file.
    
    Args:
        points: Points array [N, 3]
        colors: Colors array [N, 3] (0-255)
    """
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Mip-NeRF 360 dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing Mip-NeRF 360 dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--scene_name', type=str, default='scene',
                        help='Name of the scene')
    parser.add_argument('--no_colmap', action='store_true',
                        help='Skip COLMAP point cloud loading')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Image downsampling factor (1=original, 2=half size)')
    
    args = parser.parse_args()
    
    prepare_mipnerf360_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scene_name=args.scene_name,
        use_colmap_points=not args.no_colmap,
        image_downsample=args.downsample
    )

