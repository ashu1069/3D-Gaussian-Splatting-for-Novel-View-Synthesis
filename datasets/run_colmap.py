"""
COLMAP integration for extracting camera poses and intrinsics from images.

This script:
1. Runs COLMAP feature extraction, matching, and reconstruction
2. Converts COLMAP output to the format expected by the training script
3. Extracts camera poses (c2w matrices) and intrinsics

Requirements:
- COLMAP installed and in PATH
- Images in a directory structure
"""

import subprocess
import numpy as np
import struct
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import argparse


def check_colmap_installed():
    """Check if COLMAP is installed and accessible."""
    try:
        result = subprocess.run(['colmap', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"COLMAP found: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack bytes from file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP cameras.bin file.
    
    Returns:
        Dictionary mapping camera_id to camera parameters
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = camera_properties[4]
            
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            # COLMAP camera models:
            # 1 = SIMPLE_PINHOLE: f, cx, cy
            # 2 = PINHOLE: fx, fy, cx, cy
            # 3 = SIMPLE_RADIAL: f, cx, cy, k
            # 4 = RADIAL: f, cx, cy, k1, k2
            # 5 = OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    
    return cameras


def read_images_binary(path_to_model_file):
    """
    Read COLMAP images.bin file.
    
    Returns:
        Dictionary mapping image_id to image data
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            }
    
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def colmap_to_c2w(qvec, tvec):
    """
    Convert COLMAP camera pose (quaternion + translation) to camera-to-world matrix.
    
    COLMAP uses: w2c (world-to-camera)
    We need: c2w (camera-to-world)
    """
    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)
    
    # COLMAP: w2c = [R | t]
    # We need: c2w = [R^T | -R^T @ t]
    w2c = np.hstack([R, t])
    w2c = np.vstack([w2c, [0, 0, 0, 1]])
    
    # Convert to c2w
    c2w = np.linalg.inv(w2c)
    
    return c2w


def extract_camera_params(cameras, camera_id):
    """
    Extract camera intrinsics from COLMAP camera model.
    
    Returns:
        fx, fy, cx, cy
    """
    cam = cameras[camera_id]
    model_id = cam['model_id']
    params = cam['params']
    width = cam['width']
    height = cam['height']
    
    if model_id == 1:  # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model_id == 2:  # PINHOLE: fx, fy, cx, cy
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    elif model_id == 3:  # SIMPLE_RADIAL: f, cx, cy, k
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model_id == 4:  # RADIAL: f, cx, cy, k1, k2
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    elif model_id == 5:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
    else:
        # Default: use image center and estimate focal length
        fx = fy = width * 0.7
        cx = width / 2.0
        cy = height / 2.0
        print(f"Warning: Unknown camera model {model_id}, using defaults")
    
    return fx, fy, cx, cy


def run_colmap_reconstruction(image_dir, output_dir, database_path=None, sparse_dir=None):
    """
    Run COLMAP reconstruction pipeline.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for COLMAP output
        database_path: Path to COLMAP database file (optional)
        sparse_dir: Path to sparse reconstruction directory (optional)
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if database_path is None:
        database_path = output_dir / "database.db"
    if sparse_dir is None:
        sparse_dir = output_dir / "sparse" / "0"
    
    database_path = Path(database_path)
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running COLMAP reconstruction on {image_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Feature extraction
    print("\n[1/4] Extracting features...")
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "OPENCV",  # Use OpenCV model for better compatibility
        "--ImageReader.single_camera", "1",  # Assume single camera
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Feature extraction failed: {result.stderr}")
        raise RuntimeError("COLMAP feature extraction failed")
    print("✓ Features extracted")
    
    # Step 2: Feature matching
    print("\n[2/4] Matching features...")
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Feature matching failed: {result.stderr}")
        raise RuntimeError("COLMAP feature matching failed")
    print("✓ Features matched")
    
    # Step 3: Sparse reconstruction
    print("\n[3/4] Running sparse reconstruction...")
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir.parent),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Sparse reconstruction failed: {result.stderr}")
        print("This might be normal if images don't have enough overlap.")
        print("Trying to use existing reconstruction if available...")
        if not (sparse_dir / "cameras.bin").exists():
            raise RuntimeError("COLMAP sparse reconstruction failed and no existing reconstruction found")
    print("✓ Sparse reconstruction complete")
    
    return sparse_dir


def convert_colmap_to_training_format(sparse_dir, image_dir, output_dir):
    """
    Convert COLMAP output to training format.
    
    Args:
        sparse_dir: Path to COLMAP sparse reconstruction directory
        image_dir: Path to images directory
        output_dir: Output directory for training data
    """
    sparse_dir = Path(sparse_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(
            f"COLMAP reconstruction files not found in {sparse_dir}\n"
            f"Expected: cameras.bin, images.bin"
        )
    
    print(f"\nConverting COLMAP output to training format...")
    print(f"Reading cameras from: {cameras_bin}")
    print(f"Reading images from: {images_bin}")
    
    # Read COLMAP files
    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)
    
    print(f"Found {len(cameras)} camera(s) and {len(images)} image(s)")
    
    # Get list of image files
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.PNG"))
    image_names = {f.name: f for f in image_files}
    
    # Match COLMAP images to files and extract poses
    c2w_matrices = []
    camera_params_list = []
    matched_images = []
    
    # Sort images by name to match file order
    sorted_image_items = sorted(images.items(), key=lambda x: x[1]['name'])
    
    for image_id, image_data in sorted_image_items:
        image_name = image_data['name']
        if image_name not in image_names:
            print(f"Warning: COLMAP image '{image_name}' not found in image directory, skipping")
            continue
        
        # Convert COLMAP pose to c2w
        qvec = image_data['qvec']
        tvec = image_data['tvec']
        c2w = colmap_to_c2w(qvec, tvec)
        c2w_matrices.append(c2w)
        
        # Extract camera intrinsics
        camera_id = image_data['camera_id']
        fx, fy, cx, cy = extract_camera_params(cameras, camera_id)
        camera_params_list.append({
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        })
        
        matched_images.append(image_name)
    
    if len(c2w_matrices) == 0:
        raise ValueError("No images matched between COLMAP and image directory!")
    
    print(f"\nMatched {len(c2w_matrices)} images with poses")
    
    # Copy images to output directory in order
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying {len(matched_images)} images...")
    for i, image_name in enumerate(tqdm(matched_images)):
        src_path = image_names[image_name]
        dst_path = output_images_dir / f"{i:06d}.png"
        # Convert to PNG if needed
        from PIL import Image
        img = Image.open(src_path).convert('RGB')
        img.save(dst_path)
    
    # Save camera metadata (use first camera's intrinsics as reference)
    cam_meta = camera_params_list[0].copy()
    # Get image dimensions from first image
    first_img = Image.open(output_images_dir / "000000.png")
    cam_meta['width'] = first_img.width
    cam_meta['height'] = first_img.height
    
    np.save(output_dir / "cam_meta.npy", cam_meta)
    
    # Save camera poses
    poses_array = np.array(c2w_matrices, dtype=np.float32)
    np.save(output_dir / "poses.npy", poses_array)
    
    # Try to extract point cloud
    points3d_bin = sparse_dir / "points3D.bin"
    if points3d_bin.exists():
        print("\nExtracting point cloud...")
        try:
            from datasets.prepare_mipnerf360 import read_binary_point_cloud
            points, colors = read_binary_point_cloud(points3d_bin)
            
            # Save as PLY
            ply_path = output_dir / "pointcloud.ply"
            with open(ply_path, 'w') as f:
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
                for point, color in zip(points, colors):
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
            print(f"✓ Point cloud saved to {ply_path} ({len(points)} points)")
        except Exception as e:
            print(f"Warning: Could not extract point cloud: {e}")
    
    print(f"\n✓ Conversion complete!")
    print(f"  - {len(c2w_matrices)} images with poses")
    print(f"  - Camera metadata: {output_dir / 'cam_meta.npy'}")
    print(f"  - Camera poses: {output_dir / 'poses.npy'}")
    if points3d_bin.exists():
        print(f"  - Point cloud: {output_dir / 'pointcloud.ply'}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run COLMAP and convert to training format')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for training data')
    parser.add_argument('--colmap_dir', type=str, default=None,
                        help='Directory for COLMAP intermediate files (default: output_dir/colmap)')
    parser.add_argument('--skip_reconstruction', action='store_true',
                        help='Skip COLMAP reconstruction, only convert existing output')
    parser.add_argument('--sparse_dir', type=str, default=None,
                        help='Path to existing COLMAP sparse reconstruction (if skipping reconstruction)')
    
    args = parser.parse_args()
    
    # Check if COLMAP is installed
    if not args.skip_reconstruction and not check_colmap_installed():
        print("ERROR: COLMAP is not installed or not in PATH")
        print("\nTo install COLMAP:")
        print("  Ubuntu/Debian: sudo apt-get install colmap")
        print("  Or build from source: https://github.com/colmap/colmap")
        return 1
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if args.skip_reconstruction:
        if args.sparse_dir:
            sparse_dir = Path(args.sparse_dir)
        else:
            # Try to find sparse directory in output_dir
            sparse_dir = output_dir / "colmap" / "sparse" / "0"
            if not sparse_dir.exists():
                raise FileNotFoundError(
                    f"Sparse reconstruction not found. Expected at: {sparse_dir}\n"
                    f"Or specify with --sparse_dir"
                )
    else:
        # Run COLMAP reconstruction
        if args.colmap_dir:
            colmap_output_dir = Path(args.colmap_dir)
        else:
            colmap_output_dir = output_dir / "colmap"
        
        sparse_dir = run_colmap_reconstruction(
            image_dir=image_dir,
            output_dir=colmap_output_dir,
            sparse_dir=colmap_output_dir / "sparse" / "0"
        )
    
    # Convert to training format
    convert_colmap_to_training_format(
        sparse_dir=sparse_dir,
        image_dir=image_dir,
        output_dir=output_dir
    )
    
    print(f"\n✓ Ready for training!")
    print(f"  Use: python scripts/train.py --data_dir {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())

