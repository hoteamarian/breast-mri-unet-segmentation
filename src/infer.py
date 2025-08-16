import os
import argparse
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from UNet import Unet

# Config
DESIRED_PHASES = 6
new_spacing = (1.0, 1.0, 1.0)
linear_interp = sitk.sitkLinear
nearest_interp = sitk.sitkNearestNeighbor
target_shape = (128, 128)  # (H, W)
min_tumor_pixels = 200  # threshold for positive slice


def resample_image(itk_image, spacing, interpolator):
    """Resample an ITK image to the desired spacing."""
    orig_size = itk_image.GetSize()
    orig_spacing = itk_image.GetSpacing()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(itk_image)


def load_and_preprocess_patient(phases_folder, mask_path):
    """Load all phases and the segmentation mask, then preprocess them."""
    # Load and sort phase images
    phase_files = sorted(
        [f for f in os.listdir(phases_folder) if f.endswith(".nii.gz")]
    )

    volumes = []
    for pf in phase_files:
        img = sitk.ReadImage(os.path.join(phases_folder, pf))
        res_img = resample_image(img, new_spacing, linear_interp)
        arr = sitk.GetArrayFromImage(res_img).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)  # z-score normalization
        volumes.append(arr)

    # Load the mask
    m = sitk.ReadImage(mask_path)
    res_m = resample_image(m, new_spacing, nearest_interp)
    mask_arr = sitk.GetArrayFromImage(res_m).astype(np.uint8)

    # Select a slice containing a tumor
    Z = volumes[0].shape[0]
    tumor_slices = [z for z in range(Z) if mask_arr[z].sum() >= min_tumor_pixels]
    if tumor_slices:
        slice_idx = tumor_slices[0]
    else:
        slice_idx = Z // 2

    # Extract the selected slice from each phase
    slices = [vol[slice_idx] for vol in volumes]

    # Pad or truncate to match DESIRED_PHASES
    if len(slices) < DESIRED_PHASES:
        for _ in range(DESIRED_PHASES - len(slices)):
            slices.append(np.zeros_like(slices[0]))
    elif len(slices) > DESIRED_PHASES:
        slices = slices[:DESIRED_PHASES]

    # Resize to target_shape
    resized_slices = [
        resize(s, target_shape, order=1, preserve_range=True) for s in slices
    ]
    stack = np.stack(resized_slices, axis=0)  # (C, H, W)

    # Prepare and resize the mask
    mask_slice = mask_arr[slice_idx]
    mask_slice = resize(mask_slice, target_shape, order=0, preserve_range=True).astype(np.uint8)

    return stack, mask_slice


def load_model(path: str, device):
    """Load a saved UNet model from a .pth file and set it to evaluation mode."""
    model = Unet(input_channel=DESIRED_PHASES).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def dice_score(gt, pred):
    """Calculate Dice coefficient."""
    intersection = np.logical_and(gt, pred).sum()
    return (2. * intersection) / (gt.sum() + pred.sum() + 1e-8)


def iou_score(gt, pred):
    """Calculate Intersection over Union."""
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / (union + 1e-8)


def infer_and_visualize(phases_folder, mask_path, checkpoint_path, device, out_dir):
    """Run preprocessing, inference, and save the results."""
    # Preprocess patient data
    img_stack, gt_mask = load_and_preprocess_patient(phases_folder, mask_path)
    img_tensor = torch.from_numpy(img_stack).unsqueeze(0).to(device)  # (1, C, H, W)

    # Load model
    model = load_model(checkpoint_path, device)

    # Inference
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_probs = torch.sigmoid(pred_logits)[0, 0]  # (H, W)
    pred_mask = (pred_probs > 0.5).cpu().numpy().astype(np.uint8)

    img_np = img_stack[1]  # show one phase as background

    # Overlay masks
    masked_gt = np.ma.masked_where(gt_mask == 0, gt_mask)
    masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)

    patient_name = os.path.basename(os.path.normpath(phases_folder))

    # Metrics
    tp = np.logical_and(gt_mask == 1, pred_mask == 1).sum()
    fp = np.logical_and(gt_mask == 0, pred_mask == 1).sum()
    fn = np.logical_and(gt_mask == 1, pred_mask == 0).sum()

    dice_val = dice_score(gt_mask, pred_mask)
    iou_val = iou_score(gt_mask, pred_mask)

    metrics_path = os.path.join(out_dir, f"metrics_{patient_name}.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Tumor pixels in GT mask: {gt_mask.sum()}\n")
        f.write(f"Tumor pixels in Pred mask: {pred_mask.sum()}\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"Dice score: {dice_val:.4f}\n")
        f.write(f"IoU score: {iou_val:.4f}\n")

    print(f"Metrics saved to {metrics_path}")

    # Save plots
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].imshow(masked_gt, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[0].set_title('Ground Truth Overlay')

    axes[1].imshow(img_np, cmap='gray')
    axes[1].imshow(masked_pred, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title('Prediction Overlay')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"overlay_{patient_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet inference script (Docker-friendly)")
    parser.add_argument("--phases_folder", type=str, required=True,
                        help="Path to folder containing phase .nii.gz files")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to ground truth mask .nii.gz file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saving results")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    infer_and_visualize(
        args.phases_folder,
        args.mask_path,
        args.checkpoint_path,
        device,
        args.out_dir
    )
