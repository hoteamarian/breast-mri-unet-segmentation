# Breast Cancer MRI Slice-based Segmentation with UNet

## Overview
This project focuses on the segmentation of breast cancer lesions from MRI scans using a 2D slice-based approach with a custom UNet architecture.  
The dataset was built from **MAMMIA Breast Cancer MRI data**, converted into an efficient HDF5 format for fast training.  
The workflow includes preprocessing (resampling, resizing, phase normalization), balanced sampling of positive/negative slices, and model training with stability optimizations.

## Dataset Preparation

### Source
- MAMMIA Breast Cancer MRI Dataset  
- Original data stored in **DICOM / NIfTI (.nii.gz)** format  
- Multiple imaging phases (e.g., pre-contrast, post-contrast, etc.)

### Preprocessing Pipeline
- **Slice Selection**
  - Positive slices: contain tumor area ≥ 20px  
  - Negative slices: far from lesion-containing slices, with substantial breast tissue
  - Balanced ratio 1:1 positive:negative
- **Resampling & Resizing**
  - Resample to **1×1×1 mm** spacing using SimpleITK
  - Resize to **128×128 px**
- **Multi-phase Handling**
  - Each slice stored with **6 channels** (phases)
  - Missing phases padded with empty (black) channels to keep a fixed shape
- **Output Format**
  - Stored in HDF5 as:
    - Images: `(N, 6, H, W)`
    - Masks: `(N, 1, H, W)`
  - N = number of slices (positive and negative)

## Model

### Architecture
- **Custom UNet** with encoder-decoder structure
- **Input**: `(6, 128, 128)`  
- **Output**: `(1, 128, 128)` (binary mask)
- **Batch Normalization** after convolutional layers

### Loss Function
- Combined **Binary Cross-Entropy (BCE)** + **Dice Loss**

### Optimizer & Training Settings
- **Optimizer**: AdamW
- **Mixed Precision**: FP16 training with `torch.cuda.amp.GradScaler`
- **Gradient Clipping** to avoid exploding gradients
- **Learning Rate Scheduling**:
  - Initial: constant LR  
  - Updated: `ReduceLROnPlateau` (monitored on validation loss)

## Stability Improvements
During early experiments, the training encountered **NaN losses** after ~15 epochs.
This was resolved by:
- Adding **gradient clipping**
- Reducing initial GradScaler scale to avoid FP16 overflow
- Switching scheduler from **CosineAnnealingLR** to **ReduceLROnPlateau**

## Results
- Smooth, stable training after modifications
- Significant speed improvement when switching from `.npy` files in folders to a single HDF5 dataset
- Balanced positive/negative sampling improved model generalization
- Some experiment outputs are stored in the `results/` folder:
  - `.txt` with evaluation metrics
  - Overlay `.png` images showing predictions vs ground truth

## Usage with Docker

### Download dataset and pretrained model
- Download the MAMMIA dataset from https://github.com/LidiaGarrucho/MAMA-MIA
- Place the images and segmentations under your local project structure  
- Place the pretrained model checkpoint (e.g. `best_model_epoch89_dice0.9031.pth`) inside `data/models/`

### Build the Docker image
Run this inside the project root:

```bash
docker build -t unet-infer .

### Run inference with Docker
- Mount your local folders into the container and call the inference script:

docker run --rm \
    -v "/path/to/images:/data/images" \
    -v "/path/to/segmentations:/data/segmentations" \
    -v "/path/to/models:/data/models" \
    -v "/path/to/results:/results" \
    unet-infer \
    --phases_folder /data/images/<CASE_ID> \
    --mask_path /data/segmentations/expert/<CASE_ID>.nii.gz \
    --checkpoint_path /data/models/best_model_epoch89_dice0.9031.pth \
    --out_dir /results \
    --device auto

### Output
- The predicted masks will be saved under your mounted results/ folder.
- You can change --device auto to --device cpu if you don’t have CUDA available.
