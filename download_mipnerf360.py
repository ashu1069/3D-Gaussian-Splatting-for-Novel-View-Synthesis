"""
Download Mip-NeRF 360 dataset.

Mip-NeRF 360 dataset is available at:
https://jonbarron.info/mipnerf360/

This script helps download and extract the dataset.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
import argparse


def download_file(url, output_path):
    """
    Download a file using wget or curl.
    
    Args:
        url: URL to download
        output_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try wget first, then curl
    if subprocess.run(['which', 'wget'], capture_output=True).returncode == 0:
        cmd = ['wget', '--progress=bar:force', '-O', str(output_path), url]
    elif subprocess.run(['which', 'curl'], capture_output=True).returncode == 0:
        cmd = ['curl', '-L', '--progress-bar', '-o', str(output_path), url]
    else:
        raise RuntimeError("Neither wget nor curl found. Please install one of them.")
    
    print(f"Downloading {url}...")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"ERROR: Download failed with return code {result.returncode}")
        print(f"URL: {url}")
        print("\nThe dataset may need to be downloaded manually.")
        print("Please visit: https://jonbarron.info/mipnerf360/")
        print("Or check if the URL has changed.")
        return False
    
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Downloaded to {output_path}")
        return True
    else:
        print(f"ERROR: Download appears to have failed (file is empty or missing)")
        return False


def extract_zip(zip_path, output_dir):
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        output_dir: Directory to extract to
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path} to {output_dir}...")
    # Use Python's zipfile for better cross-platform support
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete!")


def extract_tar(tar_path, output_dir):
    """
    Extract tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        output_dir: Directory to extract to
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {tar_path} to {output_dir}...")
    subprocess.run(['tar', '-xzf', str(tar_path), '-C', str(output_dir)], check=True)
    print("Extraction complete!")


# Mip-NeRF 360 dataset URLs
# The dataset is packaged in two zip files:
# 1. Main scenes: 360_v2.zip (contains garden, bicycle, bonsai, counter, kitchen, room, stump)
# 2. Extra scenes: 360_extra_scenes.zip (contains flowers, treehill)

MIPNERF360_MAIN_URL = 'https://storage.googleapis.com/gresearch/refraw360/360_v2.zip'
MIPNERF360_EXTRA_URL = 'https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip'

# Scene mapping: which zip file contains which scenes
MIPNERF360_SCENES_MAIN = {
    'garden', 'bicycle', 'bonsai', 'counter', 'kitchen', 'room', 'stump'
}

MIPNERF360_SCENES_EXTRA = {
    'flowers', 'treehill'
}

MIPNERF360_ALL_SCENES = MIPNERF360_SCENES_MAIN | MIPNERF360_SCENES_EXTRA


def download_mipnerf360_scene(scene_name, download_dir='data/mipnerf360', extract=True):
    """
    Download a Mip-NeRF 360 scene.
    
    The dataset is packaged in two zip files:
    - 360_v2.zip: Contains main scenes (garden, bicycle, bonsai, counter, kitchen, room, stump)
    - 360_extra_scenes.zip: Contains extra scenes (flowers, treehill)
    
    Args:
        scene_name: Name of the scene (e.g., 'garden', 'bicycle')
        download_dir: Directory to download to
        extract: Whether to extract the zip file
    """
    if scene_name not in MIPNERF360_ALL_SCENES:
        available = ', '.join(sorted(MIPNERF360_ALL_SCENES))
        raise ValueError(
            f"Unknown scene: {scene_name}. Available scenes: {available}"
        )
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which zip file contains this scene
    if scene_name in MIPNERF360_SCENES_MAIN:
        url = MIPNERF360_MAIN_URL
        zip_filename = '360_v2.zip'
    else:  # scene_name in MIPNERF360_SCENES_EXTRA
        url = MIPNERF360_EXTRA_URL
        zip_filename = '360_extra_scenes.zip'
    
    zip_path = download_dir / zip_filename
    
    # Check if already downloaded
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"Zip file already exists: {zip_path}")
        print("Skipping download. Delete the file to re-download.")
    else:
        success = download_file(url, zip_path)
        if not success:
            print(f"\nFailed to download {scene_name}.")
            print("Please try downloading manually:")
            if scene_name in MIPNERF360_SCENES_MAIN:
                print(f"  {MIPNERF360_MAIN_URL}")
            else:
                print(f"  {MIPNERF360_EXTRA_URL}")
            print(f"\nAfter downloading, extract it to: {download_dir}")
            print(f"Then run: python prepare_mipnerf360.py --input_dir {download_dir}/{scene_name} --output_dir data/prepared/{scene_name}")
            return None
    
    # Extract if requested
    if extract:
        # Check if scene directory already exists
        scene_dir = download_dir / scene_name
        if scene_dir.exists() and any(scene_dir.iterdir()):
            print(f"Scene already extracted: {scene_dir}")
            print("Skipping extraction.")
        else:
            print(f"Extracting {zip_filename}...")
            extract_zip(zip_path, download_dir)
            
            # The zip files extract to a subdirectory, find the scene
            # Check common extraction patterns
            possible_locations = [
                download_dir / scene_name,
                download_dir / '360_v2' / scene_name,
                download_dir / '360_extra_scenes' / scene_name,
                download_dir / 'data' / scene_name,
            ]
            
            scene_dir = None
            for loc in possible_locations:
                if loc.exists() and any(loc.iterdir()):
                    scene_dir = loc
                    break
            
            if scene_dir is None:
                print(f"Warning: Could not find {scene_name} after extraction.")
                print(f"Please check the extracted contents in {download_dir}")
                print("The scene should be in one of these locations:")
                for loc in possible_locations:
                    print(f"  - {loc}")
                return None
    
    # Find the actual scene directory
    scene_dir = download_dir / scene_name
    if not scene_dir.exists():
        # Try alternative locations
        for loc in [download_dir / '360_v2' / scene_name, 
                   download_dir / '360_extra_scenes' / scene_name,
                   download_dir / 'data' / scene_name]:
            if loc.exists():
                scene_dir = loc
                break
    
    if scene_dir.exists():
        print(f"\nScene ready at: {scene_dir}")
        print(f"\nTo prepare for training, run:")
        print(f"  python prepare_mipnerf360.py --input_dir {scene_dir} --output_dir data/prepared/{scene_name}")
        return scene_dir
    else:
        print(f"Error: Scene directory not found: {scene_dir}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Mip-NeRF 360 dataset')
    parser.add_argument('--scene', type=str, default='garden',
                        choices=list(MIPNERF360_ALL_SCENES),
                        help='Scene to download')
    parser.add_argument('--download_dir', type=str, default='data/mipnerf360',
                        help='Directory to download to')
    parser.add_argument('--no_extract', action='store_true',
                        help='Do not extract zip file')
    parser.add_argument('--list_scenes', action='store_true',
                        help='List available scenes')
    
    args = parser.parse_args()
    
    if args.list_scenes:
        print("Available Mip-NeRF 360 scenes:")
        print("\nMain scenes (in 360_v2.zip):")
        for scene in sorted(MIPNERF360_SCENES_MAIN):
            print(f"  - {scene}")
        print("\nExtra scenes (in 360_extra_scenes.zip):")
        for scene in sorted(MIPNERF360_SCENES_EXTRA):
            print(f"  - {scene}")
        sys.exit(0)
    
    result = download_mipnerf360_scene(
        scene_name=args.scene,
        download_dir=args.download_dir,
        extract=not args.no_extract
    )
    
    if result is None:
        sys.exit(1)

