"""
core/renderer.py
Renders keypoints from a loaded template frame onto a black canvas.
Used by Panel 2 (template visualiser).
"""
import cv2
import numpy as np
import math

# ── Colour scheme ──────────────────────────────────────────────────────────
COL_POSE  = (255, 180,  30)   # amber
COL_HAND  = ( 30, 255, 140)   # green
COL_FACE  = ( 30, 220, 255)   # cyan
COL_LINE  = ( 80,  80,  80)   # dim connection lines

POSE_CONNECTIONS = [
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),   ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),       ("left_knee","left_ankle"),
    ("right_hip","right_knee"),     ("right_knee","right_ankle"),
    ("nose","left_eye"),            ("nose","right_eye"),
]

HAND_CONNECTIONS = [
    ("wrist","thumb_cmc"),("thumb_cmc","thumb_mcp"),("thumb_mcp","thumb_ip"),("thumb_ip","thumb_tip"),
    ("wrist","index_finger_mcp"),("index_finger_mcp","index_finger_pip"),
    ("index_finger_pip","index_finger_dip"),("index_finger_dip","index_finger_tip"),
    ("wrist","middle_finger_mcp"),("middle_finger_mcp","middle_finger_pip"),
    ("middle_finger_pip","middle_finger_dip"),("middle_finger_dip","middle_finger_tip"),
    ("wrist","ring_finger_mcp"),("ring_finger_mcp","ring_finger_pip"),
    ("ring_finger_pip","ring_finger_dip"),("ring_finger_dip","ring_finger_tip"),
    ("wrist","pinky_mcp"),("pinky_mcp","pinky_pip"),
    ("pinky_pip","pinky_dip"),("pinky_dip","pinky_tip"),
]


def _kp_px(kp_dict, key, w, h, mirror=True):
    """Convert normalised (x,y,z) to pixel coords."""
    if not kp_dict or key not in kp_dict:
        return None
    v = kp_dict[key]
    if v is None:
        return None

    if isinstance(v, dict):
        if "x" not in v or "y" not in v:
            return None
        # Flip X for mirror feel
        return int((1.0 - v["x"] if mirror else v["x"]) * w), int(v["y"] * h)

    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return int((1.0 - v[0] if mirror else v[0]) * w), int(v[1] * h)

    return None


def render_template_frame(frame_snap: dict, canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    Draw one template frame's keypoints on a black canvas.
    frame_snap = {pose, left_hand, right_hand, face}
    """
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    pose = frame_snap.get("pose") or {}
    lh   = frame_snap.get("left_hand")  or {}
    rh   = frame_snap.get("right_hand") or {}

    # ── Pose connections ──────────────────────────────────────────────
    for a, b in POSE_CONNECTIONS:
        pa = _kp_px(pose, a, canvas_w, canvas_h)
        pb = _kp_px(pose, b, canvas_w, canvas_h)
        if pa and pb:
            cv2.line(img, pa, pb, COL_LINE, 1)

    # ── Pose dots ─────────────────────────────────────────────────────
    for key in pose:
        pt = _kp_px(pose, key, canvas_w, canvas_h)
        if pt:
            cv2.circle(img, pt, 4, COL_POSE, -1)

    # ── Hand connections + dots ───────────────────────────────────────
    for hand_kp, col in [(lh, COL_HAND), (rh, COL_HAND)]:
        for a, b in HAND_CONNECTIONS:
            pa = _kp_px(hand_kp, a, canvas_w, canvas_h)
            pb = _kp_px(hand_kp, b, canvas_w, canvas_h)
            if pa and pb:
                cv2.line(img, pa, pb, COL_LINE, 1)
        for key in hand_kp:
            pt = _kp_px(hand_kp, key, canvas_w, canvas_h)
            if pt:
                cv2.circle(img, pt, 3, col, -1)

    return img


def render_live_dots(frame_snap: dict, canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    Same as render_template_frame but with a subtle glow tint for live view.
    """
    base = render_template_frame(frame_snap, canvas_w, canvas_h)
    # Add slight blue tint to distinguish live vs template
    tint = np.zeros_like(base)
    tint[:, :, 0] = 20   # slight blue channel boost
    return cv2.add(base, tint)


def render_metric_pose(template: dict, anim_phase: float, canvas_w: int, canvas_h: int) -> np.ndarray:
    """
    Synthesise an animated stick-figure from a V2 metric template (no frames).

    anim_phase: 0.0–1.0, driven by wall-clock time (loops).  One period = one
    oscillation cycle.  Driven by PanelTemplate.

    Arm groups: C1 = right arm,  C3 = left arm.
    Angles from template angle_mean:
      upper_arm_angle – shoulder→elbow direction (screen-space, 0-360°)
      forearm_angle   – elbow→wrist direction (screen-space, 0-360°)
    If these are absent (old templates) the function falls back to a neutral T-pose.
    """
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    metrics     = template.get("metrics", {})
    motion_type = template.get("motion_type", "")
    tracked     = template.get("tracked_groups", [])

    # ── Body scaffold (normalised → pixel) ───────────────────────────────
    cx       = canvas_w * 0.50
    cy       = canvas_h * 0.36
    torso_h  = canvas_h * 0.22
    sh_hw    = canvas_w * 0.15    # half shoulder width
    seg_len  = canvas_h * 0.17    # upper-arm / forearm pixel length

    r_sh  = (cx + sh_hw, cy)
    l_sh  = (cx - sh_hw, cy)
    r_hip = (cx + sh_hw * 0.65,  cy + torso_h)
    l_hip = (cx - sh_hw * 0.65,  cy + torso_h)
    head  = (cx, cy - canvas_h * 0.09)

    DIM   = (55, 55, 55)
    # Torso lines
    for a, b in [(r_sh, l_sh), (r_sh, r_hip), (l_sh, l_hip), (r_hip, l_hip)]:
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), DIM, 1)
    # Head circle
    cv2.circle(img, (int(head[0]), int(head[1])), int(canvas_h * 0.07), DIM, 1)

    osc = math.sin(anim_phase * math.pi * 2)   # −1 … +1 oscillation value

    def _draw_arm(group_id: str, shoulder_pt: tuple, is_right: bool):
        gdata      = metrics.get(group_id, {})
        angle_mean = gdata.get("angle_mean", {})
        ua_deg = angle_mean.get("upper_arm_angle", 20.0 if is_right else 160.0)
        fa_deg = angle_mean.get("forearm_angle",  270.0)

        # Oscillation offset applied to forearm angle
        osc_offset = 0.0
        if motion_type == "oscillation":
            osc_offset = osc * 35.0                      # ±35° swing
        elif motion_type == "bilateral_oscillation":
            phase_sign = 1.0 if is_right else -1.0       # mirror left/right
            osc_offset = phase_sign * osc * 35.0
        elif not motion_type:                            # e.g. clap: wrist-in pulse
            osc_offset = osc * 10.0 * (1.0 if is_right else -1.0)

        ua_rad = math.radians(ua_deg)
        fa_rad = math.radians(fa_deg + osc_offset)

        sx, sy = shoulder_pt
        ex = sx + seg_len * math.cos(ua_rad)
        ey = sy + seg_len * math.sin(ua_rad)
        wx = ex + seg_len * math.cos(fa_rad)
        wy = ey + seg_len * math.sin(fa_rad)

        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), COL_POSE, 2)
        cv2.line(img, (int(ex), int(ey)), (int(wx), int(wy)), COL_POSE, 2)
        cv2.circle(img, (int(sx), int(sy)), 4, COL_POSE, -1)
        cv2.circle(img, (int(ex), int(ey)), 4, COL_POSE, -1)
        cv2.circle(img, (int(wx), int(wy)), 5, COL_HAND, -1)

    if "C1" in tracked:
        _draw_arm("C1", r_sh, True)
    if "C3" in tracked:
        _draw_arm("C3", l_sh, False)

    return img
