from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# HandProfile: mô tả hình dáng và trạng thái từng tay
@dataclass
class HandProfile:
    side: str  # 'left' hoặc 'right'
    arm_present: bool
    hand_present: bool
    elbow_angle: Optional[float] = None
    forearm_angle: Optional[float] = None
    forearm_body_angle: Optional[float] = None
    hand_pose: Optional[str] = None
    finger_spread: Optional[float] = None
    palm_orient: Optional[float] = None
    direction: Optional[tuple[float, float]] = None
    anchor_pos: Optional[tuple[float, float]] = None

def build_hand_profile(states: dict) -> dict:
    """
    Tạo profile cho từng tay từ states trả về bởi GroupTracker.compute()
    """
    profiles = {}
    # Right hand
    c1 = states.get("C1")
    c2 = states.get("C2")
    profiles["right"] = HandProfile(
        side="right",
        arm_present=c1.present if c1 else False,
        hand_present=c2.present if c2 else False,
        elbow_angle=c1.angles.get("elbow") if c1 and c1.present else None,
        forearm_angle=c1.angles.get("forearm_angle") if c1 and c1.present else None,
        forearm_body_angle=c1.forearm_body_angle if c1 and c1.present else None,
        hand_pose=c2.hand_pose if c2 and c2.present else None,
        finger_spread=c2.finger_spread if c2 and c2.present else None,
        palm_orient=c2.angles.get("palm_orient") if c2 and c2.present else None,
        direction=c1.direction if c1 and c1.present else None,
        anchor_pos=c2.anchor_pos if c2 and c2.present else None,
    )
    # Left hand
    c3 = states.get("C3")
    c4 = states.get("C4")
    profiles["left"] = HandProfile(
        side="left",
        arm_present=c3.present if c3 else False,
        hand_present=c4.present if c4 else False,
        elbow_angle=c3.angles.get("elbow") if c3 and c3.present else None,
        forearm_angle=c3.angles.get("forearm_angle") if c3 and c3.present else None,
        forearm_body_angle=c3.forearm_body_angle if c3 and c3.present else None,
        hand_pose=c4.hand_pose if c4 and c4.present else None,
        finger_spread=c4.finger_spread if c4 and c4.present else None,
        palm_orient=c4.angles.get("palm_orient") if c4 and c4.present else None,
        direction=c3.direction if c3 and c3.present else None,
        anchor_pos=c4.anchor_pos if c4 and c4.present else None,
    )
    return profiles


_RIGHT_ARM_LMS = ["right_shoulder", "right_elbow", "right_wrist"]
_LEFT_ARM_LMS = ["left_shoulder", "left_elbow", "left_wrist"]
_TORSO_LMS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
_HEAD_LMS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "mouth_left", "mouth_right"]

DEFAULT_GROUPS: dict[str, dict] = {
    "C1": {"name": "Right Arm", "pose_lms": _RIGHT_ARM_LMS, "hand": None},
    "C2": {"name": "Right Hand", "pose_lms": ["right_wrist"], "hand": "right_hand"},
    "C3": {"name": "Left Arm", "pose_lms": _LEFT_ARM_LMS, "hand": None},
    "C4": {"name": "Left Hand", "pose_lms": ["left_wrist"], "hand": "left_hand"},
    "C5": {"name": "Torso", "pose_lms": _TORSO_LMS, "hand": None},
    "C6": {"name": "Head", "pose_lms": _HEAD_LMS, "hand": None},
}


@dataclass
class GroupState:
    group_id: str
    present: bool = False
    angles: dict[str, float] = field(default_factory=dict)
    ratios: dict[str, float] = field(default_factory=dict)
    direction: Optional[tuple[float, float]] = None
    anchor_pos: Optional[tuple[float, float]] = None
    finger_spread: Optional[float] = None
    # New geometric metrics:
    hand_pose: Optional[str] = None
    forearm_body_angle: Optional[float] = None


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def _dist3d(a: tuple, b: tuple) -> float:
    # Tuple mảng từ Mediapipe có 3 giá trị (x, y, z).
    # Z là depth - độ sâu tiệm cận so với mặt phẳng người. Rất quan trọng khi tay chìa thẳng ra phía trước che Camera
    if len(a) > 2 and len(b) > 2:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return _dist(a, b)

def _angle_3pts(a: tuple, b: tuple, c: tuple) -> float:
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag_ba = math.sqrt(bax**2 + bay**2)
    mag_bc = math.sqrt(bcx**2 + bcy**2)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


def _unit_vec(dx: float, dy: float) -> tuple[float, float]:
    mag = math.sqrt(dx**2 + dy**2)
    if mag < 1e-9:
        return (0.0, 0.0)
    return (dx / mag, dy / mag)


def _body_scale(pose: dict) -> float:
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    if not (ls and rs and lh and rh):
        return 0.3

    mid_s = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_h = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    d = _dist(mid_s, mid_h)
    return d if d > 1e-4 else 0.3

def _calc_body_angle(pose: dict) -> float:
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    if not (ls and rs and lh and rh):
        return 90.0 # Default straight UP in atan2(-y,x) where Y is down
    mid_s = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_h = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    bx = mid_s[0] - mid_h[0]
    by = mid_s[1] - mid_h[1] 
    return math.degrees(math.atan2(-by, bx)) % 360.0


def _compute_arm(
    gid: str,
    pose: dict,
    scale: float,
    body_angle: float,
    sh_key: str,
    el_key: str,
    wr_key: str,
    prev_wr: Optional[tuple],
) -> GroupState:
    sh = pose.get(sh_key)
    el = pose.get(el_key)
    wr = pose.get(wr_key)
    if not (sh and el and wr):
        return GroupState(group_id=gid, present=False)

    elbow_angle = _angle_3pts(sh, el, wr)
    vert = (sh[0], sh[1] - 0.1)
    shoulder_angle = _angle_3pts(vert, sh, el)

    arm_len = (_dist(sh, el) + _dist(el, wr)) / scale
    wrist_h = (sh[1] - wr[1]) / scale

    # Segment vector angles (absolute screen-space direction, 0-360°).
    # upper_arm_angle: direction from shoulder → elbow
    # forearm_angle:   direction from elbow → wrist
    # In MediaPipe coords y increases downward, so a forearm pointing
    # upward (wrist above elbow) gives ≈270°; pointing right gives ≈0°.
    upper_arm_angle = math.degrees(math.atan2(el[1] - sh[1], el[0] - sh[0])) % 360.0
    forearm_angle   = math.degrees(math.atan2(wr[1] - el[1], wr[0] - el[0])) % 360.0

    # Convert forearm to standard coordinate system (Y is up) to compare with body tilt
    fa_angle_std = math.degrees(math.atan2(-(wr[1] - el[1]), wr[0] - el[0])) % 360.0
    forearm_body_angle = (fa_angle_std - body_angle + 90.0) % 360.0

    direction = None
    if prev_wr:
        direction = _unit_vec(wr[0] - prev_wr[0], wr[1] - prev_wr[1])

    return GroupState(
        group_id=gid,
        present=True,
        angles={
            "elbow": elbow_angle,
            "shoulder": shoulder_angle,
            "upper_arm_angle": upper_arm_angle,
            "forearm_angle": forearm_angle,
        },
        ratios={"arm_len": arm_len, "wrist_height": wrist_h},
        direction=direction,
        anchor_pos=(wr[0], wr[1]),
        forearm_body_angle=forearm_body_angle,
    )


def _compute_hand(gid: str, hand_lms: Optional[dict], pose: dict, wrist_key: str) -> GroupState:
    wr_pose = pose.get(wrist_key)
    
    if not hand_lms:
        if not wr_pose:
            return GroupState(group_id=gid, present=False)
        return GroupState(
            group_id=gid, 
            present=True, 
            anchor_pos=(wr_pose[0], wr_pose[1]),
            hand_pose="unknown"
        )

    tips = ["index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip", "thumb_tip"]
    tip_pts = [hand_lms[t] for t in tips if t in hand_lms]
    wrist = hand_lms.get("wrist") or wr_pose

    spread = 0.0
    if wrist and len(tip_pts) >= 2:
        dists = [_dist(wrist, tp) for tp in tip_pts]
        max_d = max(dists) if dists else 0.001
        mean_d = sum(dists) / len(dists)
        std_d = math.sqrt(sum((d - mean_d) ** 2 for d in dists) / len(dists))
        spread = min(std_d / (max_d + 1e-9), 1.0)

    palm_angle = 0.0
    mid_mcp = hand_lms.get("middle_finger_mcp")
    if wrist and mid_mcp:
        dx = mid_mcp[0] - wrist[0]
        dy = mid_mcp[1] - wrist[1]
        palm_angle = math.degrees(math.atan2(dy, dx)) % 360

    hand_pose = "unknown"
    if wrist and mid_mcp and tip_pts and len(tip_pts) == 5:
        # Simplistic heuristic for hand poses using distances
        # Tính toán dựa trên không gian 3D (x, y, z) thay vì 2D để giải quyết việc hướng tay về phía trước làm rút ngắn khoảng cách hiển thị màng hình
        wrist_mcp_dist = _dist3d(wrist, mid_mcp)
        tip_dists = [_dist3d(wrist, hand_lms[t]) for t in tips if t in hand_lms]

        # If average tip distance from wrist is very small compared to wrist-mcp dist, it's a fist
        # Typically open palm: tips are > 1.5x to 2x further than MCP
        avg_tip_dist = sum(tip_dists) / len(tip_dists)

        index_tip_dist = _dist3d(wrist, hand_lms.get("index_finger_tip", wrist))

        # Ngưỡng bắt ngón tay khi chỉa thẳng vào màng hình (z depth lớn nhưng 2d distance nhỏ)
        if avg_tip_dist < wrist_mcp_dist * 1.5:
            # Check specifically if only index is up
            if index_tip_dist > wrist_mcp_dist * 1.8 and sum(tip_dists) - index_tip_dist < wrist_mcp_dist * 5:
                hand_pose = "index_up"
            else:
                hand_pose = "fist"
        else:
            hand_pose = "open_palm"

    return GroupState(
        group_id=gid,
        present=True,
        angles={"palm_orient": palm_angle},
        ratios={},
        finger_spread=spread,
        anchor_pos=(wrist[0], wrist[1]) if wrist else None,
        hand_pose=hand_pose
    )


def _compute_torso(pose: dict, scale: float) -> GroupState:
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    if not (ls and rs):
        return GroupState(group_id="C5", present=False)

    shoulder_w = _dist(ls, rs) / scale
    tilt = _angle_3pts((ls[0], ls[1] - 0.1), ls, rs)
    mid_chest = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0 + 0.1) # roughly mid chest
    
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    hip_w = _dist(lh, rh) / scale if lh and rh else shoulder_w

    return GroupState(
        group_id="C5",
        present=True,
        angles={"shoulder_tilt": tilt},
        ratios={"shoulder_width": shoulder_w, "hip_width": hip_w},
        anchor_pos=mid_chest
    )


def _compute_head(pose: dict, scale: float) -> GroupState:
    nose = pose.get("nose")
    l_eye = pose.get("left_eye")
    r_eye = pose.get("right_eye")
    l_ear = pose.get("left_ear")
    r_ear = pose.get("right_ear")

    if not nose:
        return GroupState(group_id="C6", present=False)

    angles, ratios = {}, {}

    if l_eye and r_eye:
        roll = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
        angles["head_roll"] = roll % 360

    if l_ear and r_ear:
        face_w = _dist(l_ear, r_ear) / scale
        ratios["face_width"] = face_w

    return GroupState(group_id="C6", present=True, angles=angles, ratios=ratios, anchor_pos=(nose[0], nose[1]))


class GroupTracker:
    def __init__(self, custom_groups: Optional[dict] = None):
        self._groups = custom_groups or DEFAULT_GROUPS
        self._prev_wrists: dict[str, Optional[tuple]] = {"C1": None, "C3": None}

    def compute(self, snapshot: dict) -> dict[str, GroupState]:
        pose = snapshot.get("pose") or {}
        lh = snapshot.get("left_hand") or {}
        rh = snapshot.get("right_hand") or {}
        scale = _body_scale(pose) if pose else 0.3
        body_angle = _calc_body_angle(pose)

        states: dict[str, GroupState] = {}

        s = _compute_arm("C1", pose, scale, body_angle, "right_shoulder", "right_elbow", "right_wrist", self._prev_wrists.get("C1"))
        if s.present:
            self._prev_wrists["C1"] = s.anchor_pos
        states["C1"] = s

        states["C2"] = _compute_hand("C2", rh or None, pose, "right_wrist")

        s = _compute_arm("C3", pose, scale, body_angle, "left_shoulder", "left_elbow", "left_wrist", self._prev_wrists.get("C3"))
        if s.present:
            self._prev_wrists["C3"] = s.anchor_pos
        states["C3"] = s

        states["C4"] = _compute_hand("C4", lh or None, pose, "left_wrist")
        states["C5"] = _compute_torso(pose, scale)
        states["C6"] = _compute_head(pose, scale)
        return states

    def reset(self):
        self._prev_wrists = {"C1": None, "C3": None}
