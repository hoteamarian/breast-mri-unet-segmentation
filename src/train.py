import sys
from pathlib import Path
from multiprocessing import freeze_support
import time
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# to ensure reproducibility and cuDNN efficiency
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# import the model and trainer
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow imports from src/
from UNet import Unet
from trainer import Trainer


def build_dataframe(h5_path):
    """
    Build a DataFrame of slice indices and binary labels from an HDF5 container.
    """
    records = []
    with h5py.File(h5_path, "r") as h5f:
        masks = h5f["masks"]         # shape: (N, 1, H, W)
        N = masks.shape[0]
        flat = masks[:, 0, :, :].reshape(N, -1)
        slice_sums = flat.sum(axis=1)
        labels = (slice_sums > 0).astype(int)
        for idx, lbl in enumerate(labels):
            records.append({"slice_idx": idx, "label": int(lbl)})
    return pd.DataFrame(records)


class HDF5SliceDataset(Dataset):
    """
    PyTorch Dataset for HDF5-backed 2D slices.
    Returns (img_tensor, mask_tensor, label).
    """
    def __init__(self, h5_path, df):
        self.h5_path = h5_path
        self.df = df.reset_index(drop=True)
        self._h5f = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_path, "r")
        row = self.df.iloc[idx]
        sidx = int(row["slice_idx"])
        label = int(row["label"])
        img_np = self._h5f["images"][sidx]      # (C,H,W)
        mask_np = self._h5f["masks"][sidx, 0]   # (H,W)
        img = torch.from_numpy(img_np).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        return img, mask, label


def main():
    freeze_support()
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    h5_path = project_root / "data/processed/mama-mia_selected_slices.h5"

    # Build DataFrame
    df = build_dataframe(str(h5_path))
    print(f"Total slices: {len(df)}, Pos: {df['label'].sum()}, Neg: {len(df)-df['label'].sum()}")

    # Train/Val/Test split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
    )
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Datasets & Loaders
    batch_size = 32  # increased for efficiency
    ds_kwargs = dict(batch_size=batch_size, pin_memory=True,
                     num_workers=8, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        HDF5SliceDataset(str(h5_path), train_df), shuffle=True, **ds_kwargs
    )
    val_loader = DataLoader(
        HDF5SliceDataset(str(h5_path), val_df), shuffle=False, **ds_kwargs
    )
    test_loader = DataLoader(
        HDF5SliceDataset(str(h5_path), test_df), shuffle=False, **ds_kwargs
    )

    # Device, model, optimizer, loss, scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = Unet(input_channel=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scaler = GradScaler()

    # Loss: combined BCE + Dice
    bce = nn.BCEWithLogitsLoss()
    def dice_loss(logits, target, smooth=1e-6):
        probs = torch.sigmoid(logits)
        p = probs.view(probs.size(0), -1)
        t = target.view(target.size(0), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2*inter + smooth) / (union + smooth)
        return (1 - dice).mean()
    def criterion(logits, target):
        return bce(logits, target) + dice_loss(logits, target)

    # estimate the time for I/O vs GPU before training for future optimizations
    batch = next(iter(train_loader))  # warm-up
    imgs, msks, _ = batch
    imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)

    # Data loading timing
    t0 = time.time()
    batch = next(iter(train_loader))
    load_time = time.time() - t0
    print(f"Data loading only: {load_time:.3f}s")

    # Compute timing with AMP
    imgs, msks, _ = batch
    imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast():
        out = model(imgs)
        loss = criterion(out, msks)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    comp_time = time.time() - t0
    print(f"Compute only (AMP): {comp_time:.3f}s")

    trainer = Trainer(
        model=model,
        num_epochs=30,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    trainer.train(train_loader, val_loader)
    metrics = trainer.get_metrics()

    # Save metrics
    torch.save(metrics, project_root / "metrics.pth")
    print("Training complete. Metrics and best model saved.")

if __name__ == "__main__":
    main()
