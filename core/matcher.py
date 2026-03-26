"""
core/matcher.py
DTW-based gesture matching between live snapshots and JSON templates.
Includes motion-aware matching to detect velocity patterns and oscillation.
"""
import json
import math
import os
from pathlib import Path


def _kp_to_vec(kp_dict: dict) -> list[float]:
    """Flatten keypoint dict {name:(x,y,z)} into flat float list."""
    if not kp_dict:
        return []
    return [v for name in sorted(kp_dict) for v in kp_dict[name][:2]]  # x,y only


def _mirror_snapshot(snap: dict) -> dict:
    """
    Mirror a snapshot horizontally (flip X coordinate from left ↔ right).
    This allows matching gesture with either hand.
    """
    mirrored = {}
    
    # Mirror hands
    if snap.get("left_hand") and isinstance(snap["left_hand"], dict):
        mirrored["right_hand"] = {
            k: (1.0 - v[0], v[1], v[2])
            for k, v in snap["left_hand"].items()
        }
    else:
        mirrored["right_hand"] = None
    
    if snap.get("right_hand") and isinstance(snap["right_hand"], dict):
        mirrored["left_hand"] = {
            k: (1.0 - v[0], v[1], v[2])
            for k, v in snap["right_hand"].items()
        }
    else:
        mirrored["left_hand"] = None
    
    # Keep face and pose as-is (they're symmetric enough)
    mirrored["pose"] = snap.get("pose")
    mirrored["face"] = snap.get("face")
    
    return mirrored


def _get_hand_wrist(snap: dict, side: str) -> tuple | None:
    """Extract wrist position (x, y) from hand keypoints."""
    hand_key = "left_hand" if side == "left" else "right_hand"
    hand_kp = snap.get(hand_key)
    if hand_kp and "wrist" in hand_kp:
        wrist = hand_kp["wrist"]
        return (wrist[0], wrist[1])
    return None


def _compute_velocity(pos_a: tuple, pos_b: tuple) -> tuple:
    """Compute velocity vector (dx, dy) between two positions."""
    if pos_a is None or pos_b is None:
        return (0, 0)
    return (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])


def _velocity_similarity(vel_a: tuple, vel_b: tuple) -> float:
    """Compare two velocity vectors. Returns 0-1 where 1 = identical."""
    mag_a = math.sqrt(vel_a[0]**2 + vel_a[1]**2)
    mag_b = math.sqrt(vel_b[0]**2 + vel_b[1]**2)
    if mag_a == 0 and mag_b == 0:
        return 1.0
    if mag_a == 0 or mag_b == 0:
        return 0.0
    # Dot product / magnitude product
    dot = vel_a[0]*vel_b[0] + vel_a[1]*vel_b[1]
    mags = mag_a * mag_b
    cos_angle = max(-1.0, min(1.0, dot / mags))
    return (cos_angle + 1.0) / 2.0  # Map [-1, 1] to [0, 1]


def _check_oscillation_pattern(frames: list, side: str) -> float:
    """
    Check if hand motion shows oscillation (waving pattern).
    Returns confidence score 0-1 indicating oscillation strength.
    """
    if len(frames) < 3:
        return 0.0
    
    # Extract wrist positions
    positions = []
    for snap in frames:
        pos = _get_hand_wrist(snap, side)
        if pos:
            positions.append(pos)
    
    if len(positions) < 3:
        return 0.0
    
    # Compute velocities
    velocities = []
    for i in range(len(positions) - 1):
        vel = _compute_velocity(positions[i], positions[i+1])
        velocities.append(vel)
    
    if len(velocities) < 2:
        return 0.0
    
    # Check for direction changes (oscillation)
    # Count how many times the X velocity changes sign
    x_direction_changes = 0
    for i in range(len(velocities) - 1):
        if velocities[i][0] * velocities[i+1][0] < 0:  # Sign change
            x_direction_changes += 1
    
    # Oscillation detected if there are at least 2 direction changes in X axis
    osc_score = min(1.0, x_direction_changes / 2.0)
    
    # Also check if motion magnitude is significant
    max_vel_mag = max(math.sqrt(v[0]**2 + v[1]**2) for v in velocities)
    if max_vel_mag < 0.05:  # Motion too small (< 5% of normalized space)
        return 0.0
    
    return osc_score


def _frame_distance(snap_a: dict, snap_b: dict) -> float:
    """
    Weighted Euclidean distance between two body snapshots.
    snap = {pose, left_hand, right_hand, face}
    """
    total, count = 0.0, 0

    def _part_dist(a_kp, b_kp, weight=1.0):
        nonlocal total, count
        if not a_kp or not b_kp:
            return
        keys = set(a_kp) & set(b_kp)
        if not keys:
            return
        d = sum(
            math.sqrt(sum((a_kp[k][i] - b_kp[k][i]) ** 2 for i in range(2)))
            for k in keys
        ) / len(keys)
        total += d * weight
        count += 1

    _part_dist(snap_a.get("pose"),       snap_b.get("pose"),       1.0)
    # Trong ngôn ngữ ký hiệu, hình dạng bàn tay và ngón tay rất quan trọng
    _part_dist(snap_a.get("left_hand"),  snap_b.get("left_hand"),  2.0)
    _part_dist(snap_a.get("right_hand"), snap_b.get("right_hand"), 2.0)
    _part_dist(snap_a.get("face"),       snap_b.get("face"),       0.3)

    return total / max(count, 1)


def _compute_motion_distance(snap_a: dict, snap_b: dict, snap_c: dict) -> float:
    """
    Compute distance based on motion pattern (velocity consistency).
    Uses frames A, B, C to compare velocity vectors.
    """
    vel_ab = (
        (_get_hand_wrist(snap_b, "right") or (0,0))[0] - (_get_hand_wrist(snap_a, "right") or (0,0))[0],
        (_get_hand_wrist(snap_b, "right") or (0,0))[1] - (_get_hand_wrist(snap_a, "right") or (0,0))[1]
    )
    return math.sqrt(vel_ab[0]**2 + vel_ab[1]**2)


def dtw_similarity(detected_frames: list, template_frames: list) -> float:
    """
    DTW between two sequences of snapshots.
    Returns similarity score 0-1 (1 = perfect match).
    """
    n, m = len(detected_frames), len(template_frames)
    if n == 0 or m == 0:
        return 0.0

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _frame_distance(detected_frames[i - 1], template_frames[j - 1])
            dp[i][j] = cost + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    dtw_dist = dp[n][m] / (n + m)
    return 1.0 / (1.0 + dtw_dist)


# ── Template loader ────────────────────────────────────────────────────────

def load_template(json_path: str) -> dict | None:
    """Load a gesture template JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalise frames into snapshot dicts
        frames = []
        for fr in data.get("frames", []):
            snap = {
                "pose":       fr.get("pose"),
                "left_hand":  fr.get("left_hand"),
                "right_hand": fr.get("right_hand"),
                "face":       fr.get("face"),
            }
            
            # Convert hand/pose data to {name: (x,y,z)} format
            for part in ("pose", "left_hand", "right_hand"):
                if snap[part]:
                    # Handle 2 formats:
                    # 1. {name: {x, y, z}} - standard format
                    # 2. {landmarks: [[x,y,z], ...]} - array format
                    
                    if isinstance(snap[part], dict):
                        # Check if it has "landmarks" key (array format)
                        if "landmarks" in snap[part] and isinstance(snap[part]["landmarks"], list):
                            # Convert array format to standard dict format
                            landmarks_array = snap[part]["landmarks"]
                            # Use same landmark names as detector.py HAND_NAMES
                            LANDMARK_NAMES = [
                                "wrist",
                                "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                                "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
                                "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
                                "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
                                "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
                            ]
                            snap[part] = {}
                            for i, coords in enumerate(landmarks_array[:len(LANDMARK_NAMES)]):
                                name = LANDMARK_NAMES[i]
                                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                                    snap[part][name] = (
                                        float(coords[0]),
                                        float(coords[1]),
                                        float(coords[2]) if len(coords) > 2 else 0.0
                                    )
                        else:
                            # Standard format {name: {x,y,z}}
                            snap[part] = {
                                k: (v["x"], v["y"], v.get("z", 0.0))
                                for k, v in snap[part].items()
                                if isinstance(v, dict) and "x" in v
                            }
            
            frames.append(snap)

        return {
            "id":          data.get("id", "unknown"),
            "name":        data.get("name", "Unnamed"),
            "description": data.get("description", ""),
            "output_text": data.get("output_text", ""),
            "frames":      frames,
            "path":        json_path,
        }
    except Exception as e:
        print(f"[matcher] Failed to load {json_path}: {e}")
        return None


def match_against_template(
    buffer: list,          # list of live snapshots
    template: dict,        # loaded template dict
    threshold: float = 0.82,
) -> dict:
    """
    Compare live buffer against a template with motion-aware verification.
    Automatically handles both left and right hand by comparing against
    both original and mirrored versions of the template.
    
    For static gestures (1-2 frames): Uses DTW only
    For dynamic gestures (3+ frames): Applies motion verification
    """
    # Compare against both original and mirrored (left ↔ right) versions
    dtw_score_orig = dtw_similarity(buffer, template["frames"])
    
    # Mirror template frames for right-hand matching
    mirrored_frames = [_mirror_snapshot(frame) for frame in template["frames"]]
    dtw_score_mirror = dtw_similarity(buffer, mirrored_frames)
    
    # Use the best match (handles either hand)
    dtw_score = max(dtw_score_orig, dtw_score_mirror)
    
    # Determine if gesture is static or dynamic based on template frames
    is_dynamic_gesture = len(template["frames"]) >= 3
    
    if is_dynamic_gesture:
        # Motion pattern verification for dynamic gestures
        # Check if detected motion shows oscillation (waving)
        right_hand_osc = _check_oscillation_pattern(buffer, "right")
        left_hand_osc = _check_oscillation_pattern(buffer, "left")
        
        # Combine oscillation signals with DTW score
        motion_score = 0.5 * right_hand_osc + 0.5 * left_hand_osc
        
        # Blend DTW with motion detection
        # If motion is weak, reduce the score significantly
        final_score = dtw_score * (0.7 + 0.3 * motion_score)
        motion_score_val = motion_score
    else:
        # For static gestures, use DTW score directly without motion penalty
        final_score = dtw_score
        motion_score_val = 1.0  # No motion expected for static
    
    return {
        "score": final_score,
        "matched": final_score >= threshold,
        "template": template,
        "dtw_score": dtw_score,
        "motion_score": motion_score_val
    }
