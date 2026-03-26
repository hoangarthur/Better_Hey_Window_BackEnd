"""
core/ml_motion_matcher.py
Drop-in replacement cho MotionMatcher — GestureTCN .

"""

from __future__ import annotations

import torch
import numpy as np
from collections import deque
from typing import Optional

from core.group_tracker import GroupTracker
from core.ml_features import group_state_to_vector, FEATURE_DIM
from core.ml_model import GestureTCN


class MLMotionMatcher:
    """
    Drop-in replacement cho MotionMatcher:
        matcher = MLMotionMatcher("models/gesture_model_best.pt")
        matcher.update(left_hand=..., right_hand=..., snapshot=snapshot)
        result = matcher.detect_motion()
    """

    # Temperature > 1.0 → làm mềm probabilities → tránh overconfident
    DEFAULT_TEMPERATURE = 1.5

    def __init__(self, model_path: str,
                 buffer_size: int = 60,
                 confidence_threshold: float = 0.70,
                 temperature: float = DEFAULT_TEMPERATURE):
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)

        n_classes  = checkpoint["n_classes"]
        input_dim  = checkpoint.get("input_dim", FEATURE_DIM)
        channels   = checkpoint.get("channels", 128)

        self.model = GestureTCN(
            input_dim=input_dim,
            num_classes=n_classes,
            channels=channels,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # label_map: {"0": "wave_hello", "1": "again_repeat", ...}
        self.label_map  = {int(k): v for k, v in checkpoint["label_map"].items()}
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.buffer_size          = max(10, min(120, int(buffer_size)))
        self._min_required_frames = min(10, self.buffer_size)

        self._feature_buffer  = deque(maxlen=self.buffer_size)
        self._group_tracker   = GroupTracker()
        self._last_snapshot: Optional[dict] = None
        self._missing_frames  = 0
        self.max_missing_frames = 20

        # Stability filter — giống MotionMatcher
        self._stable_gesture = None
        self._stable_hits    = 0
        self.stable_required = 2
        self.decision_margin = 0.06

        val_acc_str = f"{checkpoint['val_acc']:.3f}" if "val_acc" in checkpoint else "?"
        print(f"[MLMotionMatcher] Loaded: {n_classes} classes, "
              f"input_dim={input_dim}, val_acc={val_acc_str}, "
              f"temperature={temperature}")

    # ── MotionMatcher-compatible API ──────────────────────────────────────

    def update(self, left_hand=None, right_hand=None,
               pose_landmarks=None, snapshot: Optional[dict] = None):
        """
        snapshot: holistic snapshot dict from extract_snapshot() — per-finger angles.
        """
        has_hand = bool(left_hand or right_hand)
        if has_hand:
            self._missing_frames = 0
            if snapshot:
                self._last_snapshot = snapshot
                states = self._group_tracker.compute(snapshot)
                vec    = group_state_to_vector(states, snapshot)
                self._feature_buffer.append(vec)
        else:
            self._missing_frames += 1
            if self._missing_frames > self.max_missing_frames:
                self.clear_buffers()

    def detect_motion(self, batch_mode: bool = False) -> dict:
        min_frames = self._min_required_frames
        if len(self._feature_buffer) < min_frames:
            return self._empty_result("Not enough frames")

        seq = np.stack(self._feature_buffer)       # (T, FEATURE_DIM)
        x   = torch.tensor(seq).unsqueeze(0)       # (1, T, FEATURE_DIM)

        with torch.no_grad():
            probs = self.model.predict_proba(
                x, temperature=self.temperature
            )[0]                                   # (n_classes,)

        candidates = [
            {
                "gesture":      self.label_map[i],
                "gesture_name": self.label_map[i],
                "score":        float(probs[i]),
                "detected":     False,
            }
            for i in range(len(probs))
        ]
        candidates.sort(key=lambda c: c["score"], reverse=True)

        top1   = candidates[0]
        top2   = candidates[1] if len(candidates) > 1 else None
        margin = top1["score"] - (top2["score"] if top2 else 0.0)
        margin_ok = margin >= self.decision_margin

        # Stability filter
        if top1["score"] >= self.confidence_threshold and margin_ok:
            if top1["gesture"] == self._stable_gesture:
                self._stable_hits += 1
            else:
                self._stable_gesture = top1["gesture"]
                self._stable_hits    = 1
        else:
            self._stable_gesture = None
            self._stable_hits    = 0

        stable_ok = batch_mode or (self._stable_hits >= self.stable_required)
        detected  = (top1["score"] >= self.confidence_threshold
                     and margin_ok and stable_ok)

        if detected:
            top1["detected"] = True

        # Rejection: class "__unknown__"
        if detected and top1["gesture"] == "__unknown__":
            detected = False
            top1["detected"] = False

        return {
            "detected":     detected,
            "gesture":      top1["gesture"],
            "gesture_name": top1["gesture_name"],
            "score":        top1["score"],
            "candidates":   candidates,
            "debug": {
                "method":          "ml_tcn",
                "temperature":     self.temperature,
                "margin":          float(margin),
                "margin_ok":       margin_ok,
                "stable_hits":     self._stable_hits,
                "stable_required": self.stable_required,
                "buffer_frames":   len(self._feature_buffer),
            },
        }

    def clear_buffers(self):
        self._feature_buffer.clear()
        self._group_tracker.reset()
        self._last_snapshot   = None
        self._stable_gesture  = None
        self._stable_hits     = 0
        self._missing_frames  = 0

    def set_min_required_frames(self, n: int):
        self._min_required_frames = max(3, min(self.buffer_size, int(n)))

    def set_buffer_size(self, n: int):
        new_size = max(10, min(120, n))
        if new_size != self.buffer_size:
            buf = list(self._feature_buffer)[-new_size:]
            self._feature_buffer = deque(buf, maxlen=new_size)
            self.buffer_size = new_size
            self._min_required_frames = min(self._min_required_frames, self.buffer_size)

    def _empty_result(self, reason: str) -> dict:
        return {
            "detected": False, "gesture": "none",
            "gesture_name": "Unknown", "score": 0.0,
            "candidates": [], "debug": {"reason": reason},
        }

    # Stub properties để tương thích với main.py
    @property
    def has_static_metric_templates(self):
        return False

    @property
    def min_required_frames(self):
        return self._min_required_frames

    @property
    def gesture_manager(self):
        return _FakeGestureManager(self.label_map)


class _FakeGestureManager:
    """Minimal stub for main.py when accessing gesture_manager."""

    def __init__(self, label_map: dict):
        self.cache = {v: {"name": v} for v in label_map.values()}

    def get(self, key):
        return self.cache.get(key)

    def get_stats(self):
        return {"total_gestures": len(self.cache)}
