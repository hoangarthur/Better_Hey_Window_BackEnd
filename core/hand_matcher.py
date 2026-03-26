"""
core/hand_matcher.py
Sign language specific gesture matching with hand shape awareness.
Separates hand shape from position for better sign language recognition.
"""
import json
import math
from pathlib import Path


def _extract_hand_shape(hand_landmarks: dict) -> dict:
    """
    Extract hand shape features (relative positions of fingers).
    Returns normalized finger positions relative to wrist.
    """
    if not hand_landmarks or "wrist" not in hand_landmarks:
        return {}
    
    wrist = hand_landmarks["wrist"]
    shape = {}
    
    for name, coords in hand_landmarks.items():
        if name != "wrist":
            # Relative position: coordinates relative to wrist
            rel_x = coords[0] - wrist[0]
            rel_y = coords[1] - wrist[1]
            shape[name] = (rel_x, rel_y)
    
    return shape


def _extract_wrist_position(hand_landmarks: dict) -> tuple:
    """Extract absolute wrist position (x, y)."""
    if not hand_landmarks or "wrist" not in hand_landmarks:
        return None
    return (hand_landmarks["wrist"][0], hand_landmarks["wrist"][1])


def _extract_arm_orientation(pose_landmarks: dict, side: str) -> tuple:
    """
    Extract arm orientation vector (from elbow to wrist).
    side: 'left' or 'right'
    """
    if not pose_landmarks:
        return (0, 0)
    
    elbow_key = f"{side}_elbow"
    wrist_key = f"{side}_wrist"
    
    if elbow_key not in pose_landmarks or wrist_key not in pose_landmarks:
        return (0, 0)
    
    elbow = pose_landmarks[elbow_key]
    wrist = pose_landmarks[wrist_key]
    
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    
    # Normalize
    mag = math.sqrt(dx*dx + dy*dy)
    if mag < 0.001:
        return (0, 0)
    
    return (dx / mag, dy / mag)


def _shape_distance(shape_a: dict, shape_b: dict) -> float:
    """
    Distance between two hand shapes.
    Returns 0-1 where 0 = perfect match, 1 = completely different.
    """
    if not shape_a or not shape_b:
        return 1.0
    
    keys = set(shape_a.keys()) & set(shape_b.keys())
    if not keys:
        return 1.0
    
    total_dist = 0.0
    for key in keys:
        dx = shape_a[key][0] - shape_b[key][0]
        dy = shape_a[key][1] - shape_b[key][1]
        total_dist += math.sqrt(dx*dx + dy*dy)
    
    avg_dist = total_dist / len(keys)
    # Convert to similarity score (0-1)
    return min(1.0, avg_dist)


def _position_distance(pos_a: tuple, pos_b: tuple) -> float:
    """Distance between two wrist positions (normalized 0-1 scale)."""
    if not pos_a or not pos_b:
        return 1.0
    
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    return min(1.0, dist)


def _orientation_similarity(orient_a: tuple, orient_b: tuple) -> float:
    """
    Similarity between two orientation vectors.
    Returns 0-1 where 1 = same direction, 0 = opposite.
    """
    if not orient_a or not orient_b:
        return 1.0
    
    # Compute cosine similarity
    dot = orient_a[0] * orient_b[0] + orient_a[1] * orient_b[1]
    mag_a = math.sqrt(orient_a[0]**2 + orient_a[1]**2)
    mag_b = math.sqrt(orient_b[0]**2 + orient_b[1]**2)
    
    if mag_a < 0.001 or mag_b < 0.001:
        return 1.0
    
    cos_angle = dot / (mag_a * mag_b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # Convert to similarity (0-1)
    return (cos_angle + 1.0) / 2.0


def _mirror_hand_landmarks(hand_landmarks: dict) -> dict:
    """Mirror hand landmarks horizontally (flip X: x -> 1-x)."""
    if not hand_landmarks:
        return None
    return {
        k: (1.0 - v[0], v[1], v[2])
        for k, v in hand_landmarks.items()
    }


def _match_hand_aware_single(
    buffer: list,
    template_frame: dict,
    use_right_hand: bool = False,
    shape_weight: float = 0.8,
    position_weight: float = 0.1,
    orientation_weight: float = 0.1,
) -> dict:
    """
    Match buffer against template using one hand side.
    If use_right_hand=False, uses left_hand. If True, uses mirrored right_hand.
    """
    tpl_hand = template_frame.get("left_hand")
    if not tpl_hand:
        return {"shape": 0.0, "position": 0.0, "orientation": 0.0, "count": 0}
    
    # If template is for left hand but we want to match right hand, mirror it
    if use_right_hand:
        tpl_hand = _mirror_hand_landmarks(tpl_hand)
    
    tpl_pose = template_frame.get("pose")
    tpl_side = "right" if use_right_hand else "left"
    
    total_shape_score = 0.0
    total_position_score = 0.0
    total_orientation_score = 0.0
    count = 0
    
    for snapshot in buffer:
        # Get live data for the correct hand side
        live_hand_key = "right_hand" if use_right_hand else "left_hand"
        live_hand = snapshot.get(live_hand_key)
        live_pose = snapshot.get("pose")
        
        if not live_hand:
            continue
        
        count += 1
        
        # Hand shape matching
        tpl_shape = _extract_hand_shape(tpl_hand) if tpl_hand else {}
        live_shape = _extract_hand_shape(live_hand) if live_hand else {}
        
        shape_dist = _shape_distance(tpl_shape, live_shape)
        shape_score = 1.0 - shape_dist
        total_shape_score += shape_score
        
        # Position matching (wrist)
        tpl_pos = _extract_wrist_position(tpl_hand) if tpl_hand else None
        live_pos = _extract_wrist_position(live_hand) if live_hand else None
        
        pos_dist = _position_distance(tpl_pos, live_pos)
        position_score = 1.0 - pos_dist
        total_position_score += position_score
        
        # Arm orientation matching
        tpl_orient = _extract_arm_orientation(tpl_pose, tpl_side) if tpl_pose else (0, 0)
        live_orient = _extract_arm_orientation(live_pose, tpl_side) if live_pose else (0, 0)
        
        orientation_score = _orientation_similarity(tpl_orient, live_orient)
        total_orientation_score += orientation_score
    
    return {
        "shape": total_shape_score,
        "position": total_position_score,
        "orientation": total_orientation_score,
        "count": count,
    }


def match_against_template_hand_aware(
    buffer: list,
    template: dict,
    threshold: float = 0.85,
    shape_weight: float = 0.8,
    position_weight: float = 0.1,
    orientation_weight: float = 0.1,
) -> dict:
    """
    Hand-aware gesture matching for sign language.
    Separates hand shape, position, and arm orientation.
    Automatically tries both left and right hand.
    
    weights: how much each component contributes to final score
    - shape_weight: importance of hand shape (finger configuration)
    - position_weight: importance of wrist position in frame
    - orientation_weight: importance of arm orientation
    """
    import json
    if not buffer or not template or not template.get("frames"):
        return {"score": 0.0, "matched": False, "template": template}


    template_frame = template["frames"][0]  # Use first template frame

    # Try matching with left hand
    left_result = _match_hand_aware_single(
        buffer, template_frame, use_right_hand=False,
        shape_weight=shape_weight, position_weight=position_weight,
        orientation_weight=orientation_weight
    )

    # Try matching with right hand (mirrored template)
    right_result = _match_hand_aware_single(
        buffer, template_frame, use_right_hand=True,
        shape_weight=shape_weight, position_weight=position_weight,
        orientation_weight=orientation_weight
    )
    
    # Use whichever hand has more matches
    best_result = left_result if left_result["count"] >= right_result["count"] else right_result
    
    if best_result["count"] == 0:
        return {"score": 0.0, "matched": False, "template": template}
    
    # Average scores
    avg_shape = best_result["shape"] / best_result["count"]
    avg_position = best_result["position"] / best_result["count"]
    avg_orientation = best_result["orientation"] / best_result["count"]
    
    # Weighted combination
    final_score = (
        shape_weight * avg_shape +
        position_weight * avg_position +
        orientation_weight * avg_orientation
    )
    
    return {
        "score": final_score,
        "matched": final_score >= threshold,
        "template": template,
        "shape_score": avg_shape,
        "position_score": avg_position,
        "orientation_score": avg_orientation,
    }
