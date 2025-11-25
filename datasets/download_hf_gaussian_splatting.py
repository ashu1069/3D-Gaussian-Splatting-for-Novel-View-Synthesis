"""
Download and prepare Hugging Face Gaussian Splatting dataset.

This script downloads the Voxel51/gaussian_splatting dataset from Hugging Face
and prepares it for training (if it contains training images) or visualization.

Dataset: https://huggingface.co/datasets/Voxel51/gaussian_splatting
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import argparse

# Try to import datasets library (avoiding conflict with local datasets/ directory)
import sys
import importlib.util

HF_AVAILABLE = False
try:
    # Try importing from a specific location to avoid local directory conflict
    spec = importlib.util.find_spec("datasets")
    if spec and spec.origin and 'site-packages' in spec.origin:
        from datasets import load_dataset
        HF_AVAILABLE = True
except (ImportError, AttributeError):
    pass

# Also try huggingface_hub as alternative
HUB_AVAILABLE = False
try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HUB_AVAILABLE = True
except ImportError:
    pass


def download_hf_dataset(dataset_name="Voxel51/gaussian_splatting", output_dir="data/hf_gaussian_splatting"):
    """
    Download dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        output_dir: Directory to save the dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name} from Hugging Face...")
    
    # Try method 1: Use huggingface_hub (more reliable)
    if HUB_AVAILABLE:
        try:
            print("Using huggingface_hub to download dataset...")
            dataset_path = snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=str(output_dir))
            print(f"Downloaded to: {output_dir}")
            
            # Inspect what was downloaded
            files = list(output_dir.rglob('*'))
            print(f"\nDownloaded {len(files)} files")
            for f in files[:10]:  # Show first 10 files
                if f.is_file():
                    print(f"  {f.relative_to(output_dir)}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
            
            return output_dir
        except Exception as e:
            print(f"huggingface_hub download failed: {e}")
            if HF_AVAILABLE:
                print("Trying datasets library...")
            else:
                raise ImportError("Please install either 'datasets' or 'huggingface_hub': pip install datasets huggingface_hub")
    
    # Try method 2: Use datasets library
    if HF_AVAILABLE:
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name, split="train")
        print(f"Dataset loaded. Number of samples: {len(dataset)}")
        
        # Inspect the dataset structure
        if len(dataset) > 0:
            print("\nDataset structure:")
            print(f"Keys: {dataset[0].keys()}")
            for key in dataset[0].keys():
                print(f"  {key}: {type(dataset[0][key])}")
                if hasattr(dataset[0][key], 'shape'):
                    print(f"    Shape: {dataset[0][key].shape}")
                elif isinstance(dataset[0][key], dict):
                    print(f"    Dict keys: {dataset[0][key].keys()}")
        
        # Save dataset information
        info = {
            'num_samples': len(dataset),
            'keys': list(dataset[0].keys()) if len(dataset) > 0 else [],
        }
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Save all samples
        print("\nSaving dataset samples...")
        for i, sample in enumerate(tqdm(dataset)):
            sample_dir = output_dir / f"sample_{i:04d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save images if present
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, Image.Image):
                    img.save(sample_dir / 'image.png')
                elif hasattr(img, 'save'):
                    img.save(sample_dir / 'image.png')
                else:
                    # Try to convert to PIL Image
                    if isinstance(img, np.ndarray):
                        img_pil = Image.fromarray(img)
                        img_pil.save(sample_dir / 'image.png')
            
            # Save other data
            for key, value in sample.items():
                if key == 'image':
                    continue  # Already saved
                elif isinstance(value, (str, int, float)):
                    with open(sample_dir / f'{key}.txt', 'w') as f:
                        f.write(str(value))
                elif isinstance(value, np.ndarray):
                    np.save(sample_dir / f'{key}.npy', value)
                elif isinstance(value, dict):
                    with open(sample_dir / f'{key}.json', 'w') as f:
                        json.dump(value, f, indent=2)
        
            print(f"\nDataset saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"Error downloading with datasets library: {e}")
            raise
    else:
        raise ImportError("Please install either 'datasets' or 'huggingface_hub': pip install datasets huggingface_hub")


def prepare_for_training(dataset_dir, output_dir="data/prepared/hf_gaussian_splatting"):
    """
    Prepare the downloaded dataset for training.
    
    This function checks if the dataset contains training images and camera poses.
    If not, it will try to create a minimal training setup or inform the user.
    
    Args:
        dataset_dir: Directory containing downloaded dataset
        output_dir: Directory to save prepared training data
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    print(f"\nPreparing dataset for training...")
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    
    # Check dataset structure
    info_file = dataset_dir / 'dataset_info.json'
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        print(f"Dataset info: {info}")
    
    # Look for images in multiple possible structures:
    # 1. Scene directories with images/ subdirectory (e.g., bench/images/, basket/images/)
    # 2. Sample directories (sample_*)
    # 3. Direct images directory
    images_found = []
    
    # Method 1: Look for scene directories with images/ subdirectory
    scene_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    for scene_dir in scene_dirs:
        images_subdir = scene_dir / 'images'
        if images_subdir.exists() and images_subdir.is_dir():
            print(f"Found images in scene: {scene_dir.name}/images/")
            for img_file in images_subdir.glob('*.png'):
                images_found.append(img_file)
            for img_file in images_subdir.glob('*.jpg'):
                images_found.append(img_file)
            for img_file in images_subdir.glob('*.jpeg'):
                images_found.append(img_file)
    
    # Method 2: Look for sample directories
    if len(images_found) == 0:
        sample_dirs = sorted(dataset_dir.glob('sample_*'))
        for sample_dir in sample_dirs:
            for img_file in sample_dir.glob('*.png'):
                images_found.append(img_file)
            for img_file in sample_dir.glob('*.jpg'):
                images_found.append(img_file)
            for img_file in sample_dir.glob('*.jpeg'):
                images_found.append(img_file)
    
    # Method 3: Look for direct images directory
    if len(images_found) == 0:
        direct_images_dir = dataset_dir / 'images'
        if direct_images_dir.exists():
            for img_file in direct_images_dir.glob('*.png'):
                images_found.append(img_file)
            for img_file in direct_images_dir.glob('*.jpg'):
                images_found.append(img_file)
            for img_file in direct_images_dir.glob('*.jpeg'):
                images_found.append(img_file)
    
    # Method 4: Look for images anywhere in the directory tree
    if len(images_found) == 0:
        for img_file in dataset_dir.rglob('*.png'):
            images_found.append(img_file)
        for img_file in dataset_dir.rglob('*.jpg'):
            images_found.append(img_file)
        for img_file in dataset_dir.rglob('*.jpeg'):
            images_found.append(img_file)
    
    print(f"\nFound {len(images_found)} images")
    
    if len(images_found) == 0:
        print("\nWARNING: No training images found in the dataset!")
        print("The Hugging Face dataset appears to contain pre-trained Gaussian Splatting outputs")
        print("(PLY files) rather than raw training images.")
        print("\nFor training, you need:")
        print("  - Multiple images from different viewpoints")
        print("  - Camera poses (c2w matrices)")
        print("  - Camera intrinsics (fx, fy, cx, cy)")
        print("\nThis dataset may be better suited for visualization/testing pre-trained models.")
        return None
    
    # Copy images to images directory
    print(f"\nCopying {len(images_found)} images...")
    for i, img_path in enumerate(tqdm(images_found)):
        # Copy with sequential naming
        dest_path = images_dir / f"{i:06d}.png"
        shutil.copy2(img_path, dest_path)
    
    # Try to extract or estimate camera parameters
    # For now, we'll create default camera parameters
    # In a real scenario, you'd need camera poses from COLMAP or similar
    
    # Get image dimensions from first image
    if len(images_found) > 0:
        first_img = Image.open(images_found[0])
        img_width, img_height = first_img.size
        
        # Estimate camera intrinsics (default values)
        # These should ideally come from the dataset or be estimated
        fx = fy = img_width * 0.7  # Rough estimate
        cx = img_width / 2.0
        cy = img_height / 2.0
        
        cam_meta = {
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'width': int(img_width),
            'height': int(img_height),
        }
        
        # Save camera metadata
        np.save(output_dir / 'cam_meta.npy', cam_meta)
        print(f"\nCamera metadata saved:")
        print(f"  Resolution: {img_width}x{img_height}")
        print(f"  Focal length: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
        
        # Create default camera poses (identity matrices)
        # In a real scenario, these should come from COLMAP or the dataset
        num_images = len(images_found)
        poses = []
        for i in range(num_images):
            # Default: camera at origin looking down -z
            # This is a placeholder - real poses should be provided
            c2w = np.eye(4, dtype=np.float32)
            poses.append(c2w)
        
        poses_array = np.array(poses)
        np.save(output_dir / 'poses.npy', poses_array)
        print(f"\nCreated {num_images} default camera poses (identity matrices)")
        print("WARNING: These are placeholder poses. For proper training, you need real camera poses!")
        
        print(f"\n✓ Dataset prepared at: {output_dir}")
        print(f"  - {num_images} images in {images_dir}")
        print(f"  - Camera metadata: cam_meta.npy")
        print(f"  - Camera poses: poses.npy (placeholder)")
        print("\nNOTE: The camera poses are placeholders. For best results, you should:")
        print("  1. Run COLMAP on the images to get real camera poses")
        print("  2. Or use a dataset that includes camera poses")
        
        return output_dir
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and prepare Hugging Face Gaussian Splatting dataset')
    parser.add_argument('--dataset_name', type=str, default='Voxel51/gaussian_splatting',
                        help='Hugging Face dataset name')
    parser.add_argument('--download_dir', type=str, default='data/hf_gaussian_splatting',
                        help='Directory to download dataset')
    parser.add_argument('--output_dir', type=str, default='data/prepared/hf_gaussian_splatting',
                        help='Directory for prepared training data')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download and only prepare existing dataset')
    
    args = parser.parse_args()
    
    if not args.skip_download:
        # Download dataset
        dataset_dir = download_hf_dataset(args.dataset_name, args.download_dir)
    else:
        dataset_dir = Path(args.download_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Prepare for training
    prepared_dir = prepare_for_training(dataset_dir, args.output_dir)
    
    if prepared_dir:
        print(f"\n✓ Dataset ready for training!")
        print(f"  Use: python scripts/train.py --data_dir {prepared_dir}")
    else:
        print("\n✗ Dataset preparation incomplete. Check warnings above.")

