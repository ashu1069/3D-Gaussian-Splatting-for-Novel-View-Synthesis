#!/bin/bash
# Setup script for Mip-NeRF 360 dataset

SCENE=${1:-garden}
DOWNLOAD_DIR="data/mipnerf360"
PREPARED_DIR="data/prepared/${SCENE}"

echo "=========================================="
echo "Setting up Mip-NeRF 360 dataset: ${SCENE}"
echo "=========================================="

# Step 1: Download dataset
echo ""
echo "Step 1: Downloading dataset..."
if python datasets/download_mipnerf360.py --scene ${SCENE} --download_dir ${DOWNLOAD_DIR}; then
    echo "Download successful!"
else
    echo ""
    echo "=========================================="
    echo "Download failed!"
    echo "=========================================="
    echo ""
    echo "The automatic download failed (likely 404 error)."
    echo "Please download the dataset manually:"
    echo ""
    echo "1. Visit: https://jonbarron.info/mipnerf360/"
    echo "2. Download ${SCENE}.tar.gz"
    echo "3. Extract it:"
    echo "   mkdir -p ${DOWNLOAD_DIR}"
    echo "   tar -xzf ${SCENE}.tar.gz -C ${DOWNLOAD_DIR}/"
    echo ""
    echo "Then run this script again, or run:"
    echo "   python datasets/prepare_mipnerf360.py --input_dir ${DOWNLOAD_DIR}/${SCENE} --output_dir ${PREPARED_DIR} --scene_name ${SCENE}"
    echo ""
    exit 1
fi

# Step 2: Prepare dataset for training
echo ""
echo "Step 2: Preparing dataset for training..."
if python datasets/prepare_mipnerf360.py \
    --input_dir ${DOWNLOAD_DIR}/${SCENE} \
    --output_dir ${PREPARED_DIR} \
    --scene_name ${SCENE} \
    --downsample 1; then
    echo ""
    echo "=========================================="
    echo "Dataset setup complete!"
    echo "=========================================="
    echo ""
    echo "Prepared dataset location: ${PREPARED_DIR}"
    echo ""
    echo "To start training, run:"
    echo "  python scripts/train.py --data_dir ${PREPARED_DIR} --output_dir output/${SCENE} --pcd_path ${PREPARED_DIR}/pointcloud.ply"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Preparation failed!"
    echo "=========================================="
    echo ""
    echo "Please check that the dataset was extracted correctly."
    echo "Expected location: ${DOWNLOAD_DIR}/${SCENE}"
    echo ""
    exit 1
fi

