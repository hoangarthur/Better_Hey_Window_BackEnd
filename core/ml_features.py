"""
core/ml_features.py
Per-frame feature extraction for the ML gesture recognition pipeline.

Feature layout (FEATURE_DIM = 110):
  [0:72]   Base GroupState features — 6 groups × 12 features
  [72:91]  Right hand (C2) per-finger angles — 15 flexion + 4 abduction
  [91:110] Left hand  (C4) per-finger angles — 15 flexion + 4 abduction

Finger angle block (19 values per hand, normalized / 180.0 → [0, 1]):
  Flexion [0:15]:   5 fingers × 3 joints mỗi ngón = 15 góc uốn
  Abduction [15:19]: 4 cặp ngón liền kề = 4 góc spread
"""

from __future__ import annotations

import math
import numpy as np

FEATURE_DIM      = 110  # 72 base + 19 right-hand + 19 left-hand
FINGER_ANGLE_DIM = 19   # 15 flexion + 4 abduction

GROUP_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]

# Tên landmark theo thứ tự MediaPipe (khớp với detector.HAND_NAMES)
_HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Chuỗi khớp mỗi ngón: 5 điểm → 3 góc uốn (tại 3 khớp giữa)
_FINGER_CHAINS = [
    (0, 1, 2, 3, 4),      # Thumb:  wrist→cmc→mcp→ip→tip
    (0, 5, 6, 7, 8),      # Index:  wrist→mcp→pip→dip→tip
    (0, 9, 10, 11, 12),   # Middle: wrist→mcp→pip→dip→tip
    (0, 13, 14, 15, 16),  # Ring:   wrist→mcp→pip→dip→tip
    (0, 17, 18, 19, 20),  # Pinky:  wrist→mcp→pip→dip→tip
]


def _vec3(a: tuple, b: tuple) -> tuple:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _angle_between(v1: tuple, v2: tuple) -> float:
    dot  = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2)))))


def hand_lms_to_finger_angles(hand_lms: dict | None) -> list[float]:
    """
    Tính 19 đặc trưng góc ngón tay từ dict landmark của một bàn tay.

    Args:
        hand_lms: dict keyed by HAND_NAMES (từ snapshot["right_hand"] / ["left_hand"])
                  Mỗi value là tuple (x, y, z).

    Returns:
        list[float] độ dài 19, normalized [0, 1]:
          [0:15]  15 flexion angles  — 5 ngón × 3 khớp
          [15:19] 4  abduction angles — spread giữa 4 cặp ngón liền kề
        Trả về all-zeros nếu hand_lms là None hoặc thiếu landmark.
    """
    zeros = [0.0] * FINGER_ANGLE_DIM
    if not hand_lms:
        return zeros

    # Xây danh sách có thứ tự — nếu thiếu bất kỳ landmark nào thì trả zeros
    pts: list[tuple] = []
    for name in _HAND_NAMES:
        p = hand_lms.get(name)
        if p is None:
            return zeros
        pts.append(p if len(p) >= 3 else (p[0], p[1], 0.0))

    angles: list[float] = []

    # 1. Flexion angles: 3 góc uốn mỗi ngón → 15 tổng
    for chain in _FINGER_CHAINS:
        for i in range(len(chain) - 2):
            p1, p2, p3 = pts[chain[i]], pts[chain[i + 1]], pts[chain[i + 2]]
            v1 = _vec3(p2, p1)   # khớp → đầu gần
            v2 = _vec3(p2, p3)   # khớp → đầu xa
            angles.append(_angle_between(v1, v2) / 180.0)

    # 2. Abduction (spread) angles: góc giữa vector xương gốc các ngón liền kề → 4 tổng
    basal = [
        _vec3(pts[2],  pts[3]),   # Thumb  mcp→ip
        _vec3(pts[5],  pts[6]),   # Index  mcp→pip
        _vec3(pts[9],  pts[10]),  # Middle mcp→pip
        _vec3(pts[13], pts[14]),  # Ring   mcp→pip
        _vec3(pts[17], pts[18]),  # Pinky  mcp→pip
    ]
    for i in range(len(basal) - 1):
        angles.append(_angle_between(basal[i], basal[i + 1]) / 180.0)

    return angles  # length = FINGER_ANGLE_DIM = 19


def group_state_to_vector(states: dict, snapshot: dict | None = None) -> np.ndarray:
    """
    Chuyển một frame GroupStates thành vector float32 phẳng.
    Shape: (110,)

    Base layout mỗi group (12 values):
      [0]  elbow_angle         / 180.0
      [1]  forearm_angle       / 360.0
      [2]  palm_orient         / 360.0
      [3]  wrist_height        (normalized by body scale)
      [4]  arm_len             (normalized by body scale)
      [5]  direction_x         (-1 to 1)
      [6]  direction_y         (-1 to 1)
      [7]  finger_spread       (0 to 1)
      [8]  present             (0 or 1)
      [9]  hand_pose_open      (0 or 1)
      [10] hand_pose_fist      (0 or 1)
      [11] hand_pose_index_up  (0 or 1)

    Appended finger-angle blocks:
      [72:91]  right hand: 15 flexion + 4 abduction
      [91:110] left hand:  15 flexion + 4 abduction
    """
    features: list[float] = []
    for gid in GROUP_IDS:
        gs = states.get(gid)
        if not gs or not gs.present:
            features.extend([0.0] * 12)
            continue
        features.extend([
            gs.angles.get("elbow", 0.0)        / 180.0,
            gs.angles.get("forearm_angle", 0.0) / 360.0,
            gs.angles.get("palm_orient", 0.0)   / 360.0,
            gs.ratios.get("wrist_height", 0.0),
            gs.ratios.get("arm_len", 0.0),
            gs.direction[0] if gs.direction else 0.0,
            gs.direction[1] if gs.direction else 0.0,
            gs.finger_spread or 0.0,
            float(gs.present),
            float(gs.hand_pose == "open_palm") if gs.hand_pose else 0.0,
            float(gs.hand_pose == "fist")       if gs.hand_pose else 0.0,
            float(gs.hand_pose == "index_up")   if gs.hand_pose else 0.0,
        ])

    # Finger-angle blocks từ raw landmarks
    right_lms = snapshot.get("right_hand") if snapshot else None
    left_lms  = snapshot.get("left_hand")  if snapshot else None
    features.extend(hand_lms_to_finger_angles(right_lms))  # [72:91]
    features.extend(hand_lms_to_finger_angles(left_lms))   # [91:110]

    return np.array(features, dtype=np.float32)


def sequence_to_fixed_length(seq: np.ndarray, target_len: int = 60,
                              pad_value: float = 0.0) -> np.ndarray:
    """
    Pad hoặc trim sequence về fixed length.
    seq shape: (T, 110) → output shape: (target_len, 110)

    Trim: lấy đoạn giữa (bỏ frames đầu/cuối — thường là noise)
    Pad:  zeros ở đầu (causal padding)
    """
    T = len(seq)
    if T == target_len:
        return seq
    if T > target_len:
        start = (T - target_len) // 2
        return seq[start:start + target_len]
    pad = np.full((target_len - T, seq.shape[1]), pad_value, dtype=np.float32)
    return np.vstack([pad, seq])
