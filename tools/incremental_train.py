"""
tools/incremental_train.py
Fine-tune an existing model with new clips without full retraining.

Workflow:
  1. Add new clips to clips_new/<gesture_name>/
  2. Run python tools/incremental_train.py
  3. The script extracts new clips, merges dataset, and fine-tunes from checkpoint
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_model import GestureTCN
from tools.extract_dataset import _pad_sequence, extract_clip
from tools.train import AugmentedDataset, GestureDataset


def incremental_update(
    new_clips_dir: str,
    existing_dataset: str,
    checkpoint_path: str,
    output_dir: str = "models/",
    finetune_epochs: int = 30,
) -> None:
    """
    Fine-tune the model with new clips (typically 20-30 epochs).

    If a new clip belongs to a class not present in the current model,
    this function stops and asks for full retraining.
    """
    data = np.load(existing_dataset)
    x_old, y_old = data["X"], data["y"]

    labels_path = existing_dataset.replace(".npz", "_labels.json")
    with open(labels_path, encoding="utf-8") as f:
        label_map = json.load(f)
    label_map_inv = {v: int(k) for k, v in label_map.items()}

    max_len = x_old.shape[1]
    x_new: list[np.ndarray] = []
    y_new: list[int] = []

    for gesture_dir in Path(new_clips_dir).iterdir():
        if not gesture_dir.is_dir():
            continue
        gesture_name = gesture_dir.name
        if gesture_name not in label_map_inv:
            print(f"[WARN] Gesture '{gesture_name}' is not in the current model.")
            print("       Run full training instead of incremental update.")
            return

        label_idx = label_map_inv[gesture_name]
        for clip_path in sorted(gesture_dir.glob("*.mp4")):
            seq = extract_clip(str(clip_path))
            if seq is not None:
                x_new.append(_pad_sequence(seq, max_len))
                y_new.append(label_idx)
                print(f"  + {clip_path.name} ({gesture_name})")

    if not x_new:
        print("No new clips were found.")
        return

    x_combined = np.vstack([x_old, np.stack(x_new)])
    y_combined = np.hstack([y_old, np.array(y_new)])
    print(f"\nDataset: {len(x_old)} old + {len(x_new)} new = {len(x_combined)} total")

    actual_dim = x_combined.shape[2]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_classes = checkpoint["n_classes"]
    model = GestureTCN(input_dim=actual_dim, num_classes=n_classes, channels=128)
    model.load_state_dict(checkpoint["model_state"])

    aug_ds = AugmentedDataset(GestureDataset(x_combined, y_combined))
    loader = DataLoader(aug_ds, batch_size=32, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for epoch in range(1, finetune_epochs + 1):
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (model(xb).argmax(1) == yb).sum().item()

        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch}/{finetune_epochs}  "
                f"loss={total_loss/len(aug_ds):.4f}  "
                f"acc={correct/len(aug_ds):.3f}"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{output_dir}/gesture_model_best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_map": label_map,
            "n_classes": n_classes,
            "input_dim": actual_dim,
            "channels": 128,
        },
        out_path,
    )

    # Persist merged dataset.
    np.savez(
        existing_dataset,
        X=x_combined,
        y=y_combined,
        lengths=np.array([max_len] * len(x_combined)),
        max_len=max_len,
    )

    print(f"\nModel updated: {out_path}")


if __name__ == "__main__":
    incremental_update(
        new_clips_dir="clips_new/",
        existing_dataset="data/gesture_dataset.npz",
        checkpoint_path="models/gesture_model_best.pt",
    )
