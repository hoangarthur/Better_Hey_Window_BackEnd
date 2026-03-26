"""
tools/evaluate.py
Evaluate a trained model and print classification report + confusion analysis.

Run:
    python tools/evaluate.py
    python tools/evaluate.py models/gesture_model_best.pt data/gesture_dataset.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_model import GestureTCN


def evaluate(model_path: str, dataset_path: str) -> None:
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("Install scikit-learn first: pip install scikit-learn")
        return

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    label_map = {int(k): v for k, v in checkpoint["label_map"].items()}
    n_classes = checkpoint["n_classes"]
    actual_dim = checkpoint.get("input_dim", 72)

    model = GestureTCN(input_dim=actual_dim, num_classes=n_classes, channels=128)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data = np.load(dataset_path)
    x, y = data["X"], data["y"]
    x_t = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        preds = model(x_t).argmax(1).numpy()

    names = [label_map[i] for i in range(n_classes)]
    print(classification_report(y, preds, target_names=names, zero_division=0))

    # Confusion matrix: find the most common misclassifications.
    cm = confusion_matrix(y, preds)
    confusions = []
    for true_i in range(n_classes):
        for pred_i in range(n_classes):
            if true_i != pred_i and cm[true_i, pred_i] > 0:
                confusions.append((cm[true_i, pred_i], true_i, pred_i))

    if confusions:
        confusions.sort(reverse=True)
        print("Top confusions (A misclassified as B):")
        for count, ti, pi in confusions[:15]:
            print(f"  {label_map[ti]:25s} -> {label_map[pi]:25s}: {count} times")
    else:
        print("No confusions found. If val_acc is unusually high, check for overfitting.")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/gesture_model_best.pt"
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "data/gesture_dataset.npz"
    evaluate(model_path, dataset_path)
