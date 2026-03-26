"""
tools/extract_dataset.py
Extract feature sequences from video clips and save them to dataset.npz.

Expected clips directory:
    clips/
      wave_hello/
        p01_speed_normal_001.mp4
        p01_speed_fast_001.mp4
      again_repeat/
        p01_speed_normal_001.mp4
      __unknown__/
        random_001.mp4

Run:
    python tools/extract_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Add project root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import HolisticDetector, extract_snapshot
from core.group_tracker import GroupTracker
from core.ml_features import group_state_to_vector


def extract_clip(video_path: str, max_frames: int = 120) -> np.ndarray | None:
    """
    Extract one feature sequence from a video clip.

    Returns:
        np.ndarray with shape (T, FEATURE_DIM), or None if there are too few valid frames.
    """
    cap = cv2.VideoCapture(video_path)
    detector = HolisticDetector(skip_frames=1, min_hand_confidence=0.25)
    tracker = GroupTracker()
    frames: list[np.ndarray] = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        mp_results = detector.process(frame)
        has_hand = bool(mp_results.get("left_hand_landmarks") or mp_results.get("right_hand_landmarks"))

        if not has_hand:
            # If enough frames were collected and hands disappear, end this sample.
            if len(frames) > 5:
                break
            continue

        snapshot = extract_snapshot(mp_results)
        states = tracker.compute(snapshot)
        vec = group_state_to_vector(states, snapshot)
        frames.append(vec)

    cap.release()
    detector.close()

    if len(frames) < 8:
        print(f"  [SKIP] {video_path} - only {len(frames)} valid frames")
        return None

    return np.stack(frames)


def build_dataset(clips_dir: str, output_path: str) -> None:
    """Build dataset.npz from all clips in clips_dir."""
    clips_root = Path(clips_dir)
    gesture_dirs = sorted([d for d in clips_root.iterdir() if d.is_dir()])

    label_map = {i: d.name for i, d in enumerate(gesture_dirs)}
    label_map_inv = {d.name: i for i, d in enumerate(gesture_dirs)}

    print(f"Found {len(gesture_dirs)} gesture classes:")
    for i, d in enumerate(gesture_dirs):
        clips = list(d.glob("*.mp4")) + list(d.glob("*.avi"))
        flag = "  [low data]" if len(clips) < 20 else ""
        print(f"  [{i:2d}] {d.name:30s} - {len(clips)} clips{flag}")

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    lengths: list[int] = []

    for gesture_dir in gesture_dirs:
        label_idx = label_map_inv[gesture_dir.name]
        clips = sorted(list(gesture_dir.glob("*.mp4")) + list(gesture_dir.glob("*.avi")))

        for clip_path in clips:
            print(f"  Processing: {clip_path.name}", end=" ")
            seq = extract_clip(str(clip_path))
            if seq is None:
                continue
            print(f"-> {len(seq)} frames")
            sequences.append(seq)
            labels.append(label_idx)
            lengths.append(len(seq))

    if not sequences:
        print("ERROR: no valid sequences were extracted.")
        return

    # Cap sequence length at 90 frames (~3 seconds at 30 FPS).
    max_len = min(max(lengths), 90)
    padded = np.stack([_pad_sequence(s, max_len) for s in sequences])
    labels_arr = np.array(labels, dtype=np.int64)

    np.savez(
        output_path,
        X=padded,
        y=labels_arr,
        lengths=np.array(lengths),
        max_len=max_len,
    )

    labels_path = output_path.replace(".npz", "_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved: {output_path}")
    print(f"  Total samples : {len(labels)}")
    print(f"  Shape         : {padded.shape}")
    print(f"  Max seq length: {max_len}")

    print("\nClass distribution:")
    for idx, count in sorted(Counter(labels).items()):
        flag = "  [low data]" if count < 20 else ""
        print(f"  {label_map[idx]:30s}: {count} samples{flag}")


def _pad_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    t = len(seq)
    if t >= target_len:
        start = (t - target_len) // 2
        return seq[start : start + target_len]
    pad = np.zeros((target_len - t, seq.shape[1]), dtype=np.float32)
    return np.vstack([pad, seq])


if __name__ == "__main__":
    clips_dir = sys.argv[1] if len(sys.argv) > 1 else "clips/"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/gesture_dataset.npz"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    build_dataset(clips_dir, output_path)
