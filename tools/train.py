"""
tools/train.py
Training script for GestureTCN.

Chạy:
    python tools/train.py
    python tools/train.py data/gesture_dataset.npz models/ --epochs 80

Fix so với guide gốc:
  - Bug augmentation: không dùng dataset.augment = True trên shared object
    (vì val_ds.dataset là cùng object với train_ds.dataset → val set bị augment)
    → Dùng AugmentedDataset wrapper riêng biệt thay thế.
  - Mirror logic _mirror_features() cập nhật cho FEATURE_DIM=110:
    swap thêm finger-angle blocks [72:91] ↔ [91:110].
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from core.ml_model import GestureTCN, count_params
from core.ml_features import FEATURE_DIM, FINGER_ANGLE_DIM


def _stratified_split_manual(y: np.ndarray, test_size: float = 0.20) -> tuple:
    """
    Manual stratified split for highly imbalanced datasets.
    Handles classes with only 1 sample by putting them in training.
    """
    unique_classes = np.unique(y)
    train_idx = []
    val_idx = []

    for cls in unique_classes:
        cls_mask = np.where(y == cls)[0]
        n = len(cls_mask)

        # If only 1 sample, put in training
        if n == 1:
            train_idx.extend(cls_mask)
        else:
            # Split this class
            n_val = max(1, int(n * test_size))
            rng = np.random.RandomState(42)
            rng.shuffle(cls_mask)
            val_idx.extend(cls_mask[:n_val])
            train_idx.extend(cls_mask[n_val:])

    return np.array(train_idx), np.array(val_idx)


# ── Dataset ──────────────────────────────────────────────────────────────────

class GestureDataset(Dataset):
    """Dataset gốc — KHÔNG augment. Dùng AugmentedDataset nếu cần augment."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedDataset(Dataset):
    """
    Wrapper augment trên GestureDataset.
    Tách biệt hoàn toàn khỏi plain dataset → val set KHÔNG bao giờ bị ảnh hưởng.
    """

    def __init__(self, base: GestureDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return _augment(x.clone()), y


# ── Augmentation ─────────────────────────────────────────────────────────────

def _augment(x: torch.Tensor) -> torch.Tensor:
    """Data augmentation để tăng robustness."""
    T = x.shape[0]

    # 1. Time warping nhẹ (±10% speed)
    scale = 0.9 + torch.rand(1).item() * 0.2
    new_T = max(8, int(T * scale))
    if new_T != T:
        # interpolate yêu cầu (N, C, L) → permute
        xi = x.unsqueeze(0).permute(0, 2, 1)          # (1, F, T)
        xi = nn.functional.interpolate(
            xi, size=new_T, mode="linear", align_corners=False
        ).permute(0, 2, 1).squeeze(0)                  # (new_T, F)
        if new_T < T:
            pad = torch.zeros(T - new_T, x.shape[1])
            x = torch.cat([pad, xi], dim=0)
        else:
            x = xi[:T]

    # 2. Gaussian noise nhỏ trên angles/ratios
    x = x + torch.randn_like(x) * 0.015

    # 3. Mirror (50% chance) — flip direction_x, swap C1↔C3, C2↔C4, + tay
    if torch.rand(1).item() > 0.5:
        x = _mirror_features(x)

    return x


def _mirror_features(x: torch.Tensor) -> torch.Tensor:
    """
    Mirror gesture theo chiều ngang.

    Base block [0:72] — 6 groups × 12:
      Swap C1(0:12) ↔ C3(24:36), C2(12:24) ↔ C4(36:48)
      Flip direction_x (index 5 trong mỗi group)

    Finger-angle block:
      Swap right-hand block [72:91] ↔ left-hand block [91:110]
      (Flexion angles đối xứng; abduction angles cũng đối xứng)
    """
    x = x.clone()

    # Swap arm/hand groups
    c1 = x[:, 0:12].clone();  c2 = x[:, 12:24].clone()
    c3 = x[:, 24:36].clone(); c4 = x[:, 36:48].clone()
    x[:, 0:12]  = c3;  x[:, 12:24] = c4
    x[:, 24:36] = c1;  x[:, 36:48] = c2

    # Flip direction_x (feature idx 5 trong mỗi group)
    for g in range(6):
        x[:, g * 12 + 5] *= -1

    # Swap appended finger-angle blocks
    base      = 72
    fa_dim    = FINGER_ANGLE_DIM  # 19
    rh_angles = x[:, base           : base + fa_dim].clone()
    lh_angles = x[:, base + fa_dim  : base + 2 * fa_dim].clone()
    x[:, base           : base + fa_dim]     = lh_angles
    x[:, base + fa_dim  : base + 2 * fa_dim] = rh_angles

    return x


# ── Training loop ─────────────────────────────────────────────────────────────

def train(dataset_path: str, output_dir: str = "models/",
          epochs: int = 80, lr: float = 1e-3, batch_size: int = 32):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data      = np.load(dataset_path)
    X, y      = data["X"], data["y"]
    n_classes = int(np.unique(y).shape[0])

    labels_path = dataset_path.replace(".npz", "_labels.json")
    with open(labels_path, encoding="utf-8") as f:
        label_map = json.load(f)

    actual_dim = X.shape[2]
    if actual_dim != FEATURE_DIM:
        print(f"[WARN] Dataset dim={actual_dim} ≠ FEATURE_DIM={FEATURE_DIM}. "
              f"Re-extract dataset nếu bạn đã thay đổi ml_features.py.")

    print(f"Dataset : {X.shape}, {n_classes} classes, feature_dim={actual_dim}")

    # For tiny datasets (N < 50), train on all data WITHOUT augmentation
    # Reason: too few samples + augmentation mismatch (train augmented, val not)
    use_augmentation = len(y) >= 50

    if not use_augmentation:
        print(f"WARNING: Dataset too small ({len(y)} < 50) for validation split.")
        print(f"  -> Training on full dataset WITHOUT augmentation (to avoid train/val mismatch)")
        train_idx = np.arange(len(y))
        val_idx = np.array([])
    else:
        train_idx, val_idx = _stratified_split_manual(y, test_size=0.20)

    train_y_check = y[train_idx]
    val_y_check = y[val_idx] if len(val_idx) > 0 else y[:0]
    print(f"Train: {len(train_idx)} samples, {len(np.unique(train_y_check))} classes")
    if len(val_idx) > 0:
        print(f"Val:   {len(val_idx)} samples, {len(np.unique(val_y_check))} classes")
    else:
        print(f"Val:   (full dataset training - validation=training loss, NO augmentation)")

    plain_ds   = GestureDataset(X, y)
    train_base = GestureDataset(X[train_idx], y[train_idx])
    if len(val_idx) > 0:
        val_ds = GestureDataset(X[val_idx], y[val_idx])
    else:
        # For tiny datasets, validation = training data (same as train)
        val_ds = GestureDataset(X[train_idx], y[train_idx])

    # Apply augmentation only if dataset is large enough
    if use_augmentation:
        train_aug = AugmentedDataset(train_base)
    else:
        train_aug = train_base  # No augmentation for tiny dataset

    train_dl = DataLoader(train_aug, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,    batch_size=batch_size,
                          shuffle=False, num_workers=0)

    # Class-weighted loss nếu mất cân bằng
    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    weights      = 1.0 / (class_counts + 1e-6)
    weights      = torch.tensor(weights / weights.sum(), dtype=torch.float32)

    model     = GestureTCN(input_dim=actual_dim, num_classes=n_classes, channels=128)
    print(f"Model   : {count_params(model)}")
    aug_status = "WITH data augmentation" if use_augmentation else "WITHOUT data augmentation (tiny dataset)"
    print(f"Training: {aug_status}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_acc = -1.0
    n_train, n_val_ = len(train_idx), len(val_idx) if len(val_idx) > 0 else len(train_idx)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        # Recalculate train accuracy in eval mode (for fair comparison with val)
        train_correct_eval = 0
        with torch.no_grad():
            for xb, yb in train_dl:
                train_correct_eval += (model(xb).argmax(1) == yb).sum().item()

        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_correct += (model(xb).argmax(1) == yb).sum().item()

        # Use eval mode accuracy for reporting (Dropout/BatchNorm off)
        train_acc = train_correct_eval / n_train if n_train > 0 else 0.0
        val_acc   = val_correct   / n_val_  if n_val_  > 0 else 0.0
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            val_str = f"val={val_acc:.3f}" if len(val_idx) > 0 else "val=train"
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"loss={train_loss/n_train if n_train > 0 else 0.0:.4f}  "
                  f"train={train_acc:.3f}  {val_str}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label_map":   label_map,
                "n_classes":   n_classes,
                "input_dim":   actual_dim,
                "channels":    128,
                "val_acc":     val_acc,
            }, f"{output_dir}/gesture_model_best.pt")

    print(f"\nBest val accuracy : {best_val_acc:.3f}")
    print(f"Model saved       : {output_dir}/gesture_model_best.pt")


def train_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict,
    output_dir: str = "models/",
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 32,
    on_epoch=None,
) -> float:
    """
    Train GestureTCN trực tiếp từ arrays trong memory.

    Args:
        X:          (N, T, F) float32
        y:          (N,) int64
        label_map:  {int → gesture_name}
        on_epoch:   callback(epoch, total, loss, train_acc, val_acc)
                    Trả về True để dừng sớm (early stop).
    Returns:
        best_val_acc (float)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_classes  = len(label_map)
    actual_dim = X.shape[2]

    # For tiny datasets (N < 50), train on all data WITHOUT augmentation
    use_augmentation = len(y) >= 50

    if not use_augmentation:
        print(f"[train_from_arrays] Dataset too small ({len(y)} < 50) for validation split.")
        print(f"  -> Training on full dataset WITHOUT augmentation")
        train_idx = np.arange(len(y))
        val_idx = np.array([])
    else:
        train_idx, val_idx = _stratified_split_manual(y, test_size=0.20)

    train_base = GestureDataset(X[train_idx], y[train_idx])
    if len(val_idx) > 0:
        val_ds = GestureDataset(X[val_idx], y[val_idx])
    else:
        val_ds = GestureDataset(X[train_idx], y[train_idx])

    if use_augmentation:
        train_aug = AugmentedDataset(train_base)
    else:
        train_aug = train_base

    bs       = max(1, min(batch_size, len(train_idx)))
    train_dl = DataLoader(train_aug, batch_size=bs, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,    batch_size=bs, shuffle=False, num_workers=0)

    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    weights_raw  = 1.0 / (class_counts + 1e-6)
    weights      = torch.tensor(weights_raw / weights_raw.sum(), dtype=torch.float32)

    model     = GestureTCN(input_dim=actual_dim, num_classes=n_classes, channels=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_val_acc = -1.0          # -1 ensures first epoch always saves
    n_train_     = len(train_idx)
    n_val_       = len(val_idx) if len(val_idx) > 0 else len(train_idx)
    json_map     = {str(k): v for k, v in label_map.items()}
    aug_status   = "WITH data augmentation" if use_augmentation else "WITHOUT data augmentation (tiny dataset)"
    print(f"[train_from_arrays] Training: {aug_status}")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct = 0.0, 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        t_correct_eval = 0
        with torch.no_grad():
            for xb, yb in train_dl:
                t_correct_eval += (model(xb).argmax(1) == yb).sum().item()

        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                v_correct += (model(xb).argmax(1) == yb).sum().item()

        # Use eval mode accuracy for reporting
        train_acc = t_correct_eval / n_train_ if n_train_ > 0 else 0.0
        val_acc   = v_correct / n_val_   if n_val_   > 0 else 0.0
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label_map":   json_map,
                "n_classes":   n_classes,
                "input_dim":   actual_dim,
                "channels":    128,
                "val_acc":     best_val_acc,
            }, f"{output_dir}/gesture_model_best.pt")

        if on_epoch:
            stop = on_epoch(epoch, epochs, t_loss / n_train_ if n_train_ > 0 else 0.0, train_acc, val_acc)
            if stop:
                break

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",    nargs="?", default="data/gesture_dataset.npz")
    parser.add_argument("output_dir", nargs="?", default="models/")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()

    train(args.dataset, args.output_dir,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
