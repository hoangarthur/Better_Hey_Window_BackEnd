from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.group_tracker import GroupState, HandProfile


@dataclass
class GroupWindow:
    group_id: str
    present_ratio: float = 0.0
    angle_mean: dict[str, float] = field(default_factory=dict)
    angle_std: dict[str, float] = field(default_factory=dict)
    ratio_mean: dict[str, float] = field(default_factory=dict)
    ratio_std: dict[str, float] = field(default_factory=dict)
    dominant_dir: Optional[tuple[float, float]] = None
    dir_consistency: float = 0.0
    spread_mean: Optional[float] = None
    spread_std: Optional[float] = None


@dataclass
class MotionWindow:
    groups: dict[str, GroupWindow] = field(default_factory=dict)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, std


def _dir_consistency(dirs: list[tuple[float, float]]) -> float:
    if len(dirs) < 2:
        return 0.0
    sx = sum(d[0] for d in dirs)
    sy = sum(d[1] for d in dirs)
    mag = math.sqrt(sx**2 + sy**2) / len(dirs)
    return mag


def aggregate_window(state_buffer: list[dict[str, GroupState]], active_groups: Optional[list[str]] = None) -> MotionWindow:
    if not state_buffer:
        return MotionWindow()

    n = len(state_buffer)
    all_gids = set()
    for frame in state_buffer:
        all_gids.update(frame.keys())

    if active_groups:
        gids = [g for g in active_groups if g in all_gids]
    else:
        gids = sorted(all_gids)

    result = MotionWindow()

    for gid in gids:
        frames_with_group = [f[gid] for f in state_buffer if gid in f]
        present_frames = [s for s in frames_with_group if s.present]
        present_ratio = len(present_frames) / n

        angle_series: dict[str, list[float]] = {}
        ratio_series: dict[str, list[float]] = {}
        dir_list: list[tuple[float, float]] = []
        spread_list: list[float] = []

        for s in present_frames:
            for k, v in s.angles.items():
                angle_series.setdefault(k, []).append(v)
            for k, v in s.ratios.items():
                ratio_series.setdefault(k, []).append(v)
            if s.direction:
                dir_list.append(s.direction)
            if s.finger_spread is not None:
                spread_list.append(s.finger_spread)

        am, astd = {}, {}
        for k, vals in angle_series.items():
            m, s_ = _mean_std(vals)
            am[k], astd[k] = m, s_

        rm, rstd = {}, {}
        for k, vals in ratio_series.items():
            m, s_ = _mean_std(vals)
            rm[k], rstd[k] = m, s_

        dom_dir = None
        dir_con = 0.0
        if dir_list:
            sx = sum(d[0] for d in dir_list) / len(dir_list)
            sy = sum(d[1] for d in dir_list) / len(dir_list)
            mag = math.sqrt(sx**2 + sy**2)
            dom_dir = (sx / mag, sy / mag) if mag > 1e-9 else (0.0, 0.0)
            dir_con = _dir_consistency(dir_list)

        spread_m = spread_s = None
        if spread_list:
            spread_m, spread_s = _mean_std(spread_list)

        result.groups[gid] = GroupWindow(
            group_id=gid,
            present_ratio=present_ratio,
            angle_mean=am,
            angle_std=astd,
            ratio_mean=rm,
            ratio_std=rstd,
            dominant_dir=dom_dir,
            dir_consistency=dir_con,
            spread_mean=spread_m,
            spread_std=spread_s,
        )

    return result


def extract_template_metrics(template: dict) -> MotionWindow:
    raw = template.get("metrics", {})
    mw = MotionWindow()

    for gid, gdata in raw.items():
        dd = gdata.get("dominant_dir")
        # Treat [0, 0] as "no direction" so static gestures don't get penalized
        if dd and any(abs(v) > 1e-9 for v in dd):
            dom_dir = tuple(dd)
        else:
            dom_dir = None
        mw.groups[gid] = GroupWindow(
            group_id=gid,
            present_ratio=gdata.get("present_ratio", 1.0),
            angle_mean=gdata.get("angle_mean", {}),
            angle_std=gdata.get("angle_std", {}),
            ratio_mean=gdata.get("ratio_mean", {}),
            ratio_std=gdata.get("ratio_std", {}),
            dominant_dir=dom_dir,
            dir_consistency=gdata.get("dir_consistency", 0.0),
            spread_mean=gdata.get("spread_mean"),
            spread_std=gdata.get("spread_std"),
        )

    return mw


def group_window_distance(
    live: GroupWindow,
    tmpl: GroupWindow,
    weight_angle: float = 1.0,
    weight_ratio: float = 0.6,
    weight_dir: float = 0.8,
    weight_spread: float = 0.4,
    debug_dict: dict = None,
) -> float:
    total, wsum = 0.0, 0.0

    # Angles: circular distance + std tolerance (penalty is 0 within 1σ)
    common_angles = set(live.angle_mean) & set(tmpl.angle_mean)
    if common_angles:
        total_d = 0.0
        for k in common_angles:
            diff = abs(live.angle_mean[k] - tmpl.angle_mean[k]) % 360.0
            diff = min(diff, 360.0 - diff)          # circular minimum distance
            std = max(tmpl.angle_std.get(k, 45.0), 5.0)
            # 0 within 1σ, increases to 1.0 at 3σ
            total_d += min(max(diff - std, 0.0) / (2.0 * std), 1.0)
            if debug_dict is not None:
                debug_dict[k] = f"{live.angle_mean[k]:.0f}/{tmpl.angle_mean[k]:.0f}"
        avg_d = total_d / len(common_angles)
        total += avg_d * weight_angle
        if debug_dict is not None:
            debug_dict['angle_d'] = avg_d
        wsum += weight_angle

    # Ratios: std tolerance
    common_ratios = set(live.ratio_mean) & set(tmpl.ratio_mean)
    if common_ratios:
        total_d = 0.0
        for k in common_ratios:
            diff = abs(live.ratio_mean[k] - tmpl.ratio_mean[k])
            std = max(tmpl.ratio_std.get(k, 0.1), 0.01)
            total_d += min(max(diff - std, 0.0) / (2.0 * std), 1.0)
            if debug_dict is not None:
                debug_dict[k] = f"{live.ratio_mean[k]:.2f}/{tmpl.ratio_mean[k]:.2f}"
        avg_d = total_d / len(common_ratios)
        total += avg_d * weight_ratio
        if debug_dict is not None:
            debug_dict['ratio_d'] = avg_d
        wsum += weight_ratio

    # Direction: only compare when both directions have meaningful magnitude
    if live.dominant_dir and tmpl.dominant_dir:
        lx, ly = live.dominant_dir
        tx, ty = tmpl.dominant_dir
        live_mag2 = lx * lx + ly * ly
        tmpl_mag2 = tx * tx + ty * ty
        if live_mag2 > 1e-6 and tmpl_mag2 > 1e-6:
            dot = lx * tx + ly * ty
            dir_d = (1.0 - max(-1.0, min(1.0, dot))) / 2.0
            effective_w = weight_dir * max(tmpl.dir_consistency, 0.3)
            total += dir_d * effective_w
            if debug_dict is not None:
                debug_dict['dir_d'] = dir_d
                debug_dict["dir"] = f"{lx:.2f},{ly:.2f}/{tx:.2f},{ty:.2f}"
            wsum += effective_w

    # Spread: std tolerance
    if live.spread_mean is not None and tmpl.spread_mean is not None:
        diff = abs(live.spread_mean - tmpl.spread_mean)
        std = max(tmpl.spread_std or 0.08, 0.02)
        d = min(max(diff - std, 0.0) / (2.0 * std), 1.0)
        total += d * weight_spread
        if debug_dict is not None:
            debug_dict['spread_d'] = d
            debug_dict["spread"] = f"{live.spread_mean:.2f}/{tmpl.spread_mean:.2f}"
        wsum += weight_spread

    return total / wsum if wsum > 0 else 0.5


def motion_window_similarity(live_mw: MotionWindow, tmpl_mw: MotionWindow, tracked: list[str], debug_dict: dict = None) -> float:
    if not tracked:
        return 0.0

    total_dist, total_weight = 0.0, 0.0

    for gid in tracked:
        # Trong ngôn ngữ ký hiệu, tay đóng vai trò trọng tâm hơn
        group_weight = 2.0 if gid in ("lh", "rh") else 1.0

        live_gw = live_mw.groups.get(gid)
        tmpl_gw = tmpl_mw.groups.get(gid)

        if not live_gw or not tmpl_gw:
            total_dist += 1.0 * group_weight
            total_weight += group_weight
            continue

        if live_gw.present_ratio < 0.3:
            continue

        g_debug = {}
        d = group_window_distance(live_gw, tmpl_gw, debug_dict=g_debug)
        if debug_dict is not None:
            debug_dict[gid] = {"dist": d, **g_debug}
        total_dist += d * group_weight
        total_weight += group_weight

    if total_weight == 0.0:
        return 0.0

    avg_dist = total_dist / total_weight
    if debug_dict is not None:
        debug_dict['avg_dist'] = avg_dist
    # Softer mapping: short real-world motions should still produce useful confidence.
    return 1.0 / (1.0 + avg_dist * 1.2)


def hand_profile_similarity(profile_left: HandProfile, profile_right: HandProfile, template: dict, debug_dict: dict = None) -> float:
    """
    So sánh từng tay qua HandProfile với template JSON.
    Trả về score tổng hợp (0-1).
    """
    score = 1.0
    penalties = []
    # Lấy template cho từng tay
    tmpl_left = template.get("left_hand", {})
    tmpl_right = template.get("right_hand", {})
    # So sánh pose
    if tmpl_left.get("pose") and profile_left.hand_pose != tmpl_left["pose"]:
        penalties.append(0.2)
        if debug_dict is not None:
            debug_dict["left_pose"] = f"{profile_left.hand_pose} != {tmpl_left['pose']}"
    if tmpl_right.get("pose") and profile_right.hand_pose != tmpl_right["pose"]:
        penalties.append(0.2)
        if debug_dict is not None:
            debug_dict["right_pose"] = f"{profile_right.hand_pose} != {tmpl_right['pose']}"
    # So sánh spread
    if tmpl_left.get("spread") is not None and profile_left.finger_spread is not None:
        diff = abs(profile_left.finger_spread - tmpl_left["spread"])
        if diff > 0.15:
            penalties.append(0.15)
            if debug_dict is not None:
                debug_dict["left_spread"] = diff
    if tmpl_right.get("spread") is not None and profile_right.finger_spread is not None:
        diff = abs(profile_right.finger_spread - tmpl_right["spread"])
        if diff > 0.15:
            penalties.append(0.15)
            if debug_dict is not None:
                debug_dict["right_spread"] = diff
    # So sánh hướng
    if tmpl_left.get("direction") and profile_left.direction:
        tdir = tmpl_left["direction"]
        ldir = profile_left.direction
        dot = ldir[0]*tdir[0] + ldir[1]*tdir[1]
        if dot < 0.7:
            penalties.append(0.15)
            if debug_dict is not None:
                debug_dict["left_direction"] = dot
    if tmpl_right.get("direction") and profile_right.direction:
        tdir = tmpl_right["direction"]
        rdir = profile_right.direction
        dot = rdir[0]*tdir[0] + rdir[1]*tdir[1]
        if dot < 0.7:
            penalties.append(0.15)
            if debug_dict is not None:
                debug_dict["right_direction"] = dot
    # Có thể bổ sung các tiêu chí khác (elbow_angle, palm_orient, anchor_pos...)
    for p in penalties:
        score -= p
    score = max(0.0, score)
    if debug_dict is not None:
        debug_dict["final_score"] = score
    return score
