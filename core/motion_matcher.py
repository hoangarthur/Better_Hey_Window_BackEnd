"""
Motion-based gesture detection (V2 Metrics Only)
Detects dynamic gestures using cluster-based metrics and motion windows
Supports: thumbs_up, clap_hands, wave_hello variants
"""

import json
import math
import time
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any
from core.gesture_manager import GestureManager
from core.group_tracker import GroupTracker
from core.motion_metrics import GroupWindow, MotionWindow, aggregate_window, extract_template_metrics, motion_window_similarity
from core.group_tracker import build_hand_profile
from core.motion_metrics import hand_profile_similarity


class MotionMatcher:
    """
    Detect motion-based gestures using V2 cluster-based metrics.
    All gestures use GroupTracker to compute aggregate MotionWindow statistics.
    """
    EXCLUDED_ACTION_KEYS = {"thumbs_up", "thumbs_up_v2"}
    
    def __init__(self, buffer_size: int = 15, gesture_dir: str = "assets/gestures"):
        """
        Args:
            buffer_size: Number of frames to keep in history
            gesture_dir: Path to gestures directory
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = 0.55
        self.min_required_frames = max(4, buffer_size // 3)
        self.max_missing_frames = 25
        self._missing_frame_count = 0
        # Precision-oriented post filters:
        # - margin: top-1 score must exceed top-2 by this amount
        # - stability: top-1 must persist for N cycles
        self.decision_margin = 0.06
        self.stable_required_hits = 2
        self._stable_gesture = None
        self._stable_hits = 0

        # Debug logging control
        self.debug_logging_enabled = True
        self.debug_log_rejections = True
        self.debug_log_scoring = True

        # Buffers for V2 metrics pipeline
        self.hand_positions = deque(maxlen=buffer_size)
        self.hand_landmarks_buffer = deque(maxlen=buffer_size)
        self.pose_landmarks_buffer = deque(maxlen=buffer_size)
        self.group_state_buffer = deque(maxlen=buffer_size)
        self.snapshot_buffer = deque(maxlen=buffer_size)
        
        self.gesture_manager = GestureManager(gesture_dir)
        self.motion_gestures = self.gesture_manager.get_all(category="actions")
        self.group_tracker = GroupTracker()
        self.action_templates = self._load_action_templates()
        self.has_static_metric_templates = any(
            tpl.get("version") == 2 and not tpl.get("requires_motion", True)
            for tpl in self.action_templates.values()
        )

        # Motion boost mode: normalize motion by body scale and cycle timing
        self.motion_boost_enabled = False
        self._body_scale_estimated = 1.0  # Shoulder width / reference width
        self._motion_speed_avg = 0.0  # Average frame-to-frame hand movement speed
        self.gate_fail_factor_override: Optional[float] = None

        # Peak score tracker: ghi nhận điểm cao nhất từng đạt được cho mỗi template
        # {gesture_key: {"score": float, "frame_count": int, "gesture_name": str}}
        self._peak_records: dict = {}
        self._peak_threshold = 0.80  # Ngưỡng để ghi nhận peak

        if self.motion_gestures:
            print(f"[OK] Motion matcher loaded {len(self.motion_gestures)} actions (V2 metrics)")
        
        v2_count = sum(1 for t in self.action_templates.values() if t.get("version") == 2)
        if v2_count:
            print(f"[OK] Loaded {v2_count} V2 metric gesture templates")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _serialize_motion_window(self, motion_window: MotionWindow) -> dict[str, dict[str, Any]]:
        """Serialize MotionWindow to JSON-compatible dict."""
        serialized: dict[str, dict[str, Any]] = {}
        for gid, gw in motion_window.groups.items():
            serialized[gid] = {
                "present_ratio": float(gw.present_ratio),
                "angle_mean": {k: float(v) for k, v in gw.angle_mean.items()},
                "angle_std": {k: float(v) for k, v in gw.angle_std.items()},
                "ratio_mean": {k: float(v) for k, v in gw.ratio_mean.items()},
                "ratio_std": {k: float(v) for k, v in gw.ratio_std.items()},
                "dominant_dir": list(gw.dominant_dir) if gw.dominant_dir else None,
                "dir_consistency": float(gw.dir_consistency),
                "spread_mean": float(gw.spread_mean) if gw.spread_mean is not None else None,
                "spread_std": float(gw.spread_std) if gw.spread_std is not None else None,
            }
        return serialized

    def _deserialize_motion_window(self, raw_metrics: dict[str, dict[str, Any]]) -> MotionWindow:
        """Deserialize JSON metrics to MotionWindow."""
        mw = MotionWindow()
        if not isinstance(raw_metrics, dict):
            return mw

        for gid, gdata in raw_metrics.items():
            if not isinstance(gdata, dict):
                continue
            dominant_dir = gdata.get("dominant_dir")
            if isinstance(dominant_dir, list) and len(dominant_dir) == 2:
                dominant_dir = (self._safe_float(dominant_dir[0]), self._safe_float(dominant_dir[1]))
            else:
                dominant_dir = None

            gw = GroupWindow(
                group_id=gid,
                present_ratio=self._safe_float(gdata.get("present_ratio"), 0.0),
                angle_mean={k: self._safe_float(v) for k, v in (gdata.get("angle_mean") or {}).items()},
                angle_std={k: self._safe_float(v) for k, v in (gdata.get("angle_std") or {}).items()},
                ratio_mean={k: self._safe_float(v) for k, v in (gdata.get("ratio_mean") or {}).items()},
                ratio_std={k: self._safe_float(v) for k, v in (gdata.get("ratio_std") or {}).items()},
                dominant_dir=dominant_dir,
                dir_consistency=self._safe_float(gdata.get("dir_consistency"), 0.0),
                spread_mean=(self._safe_float(gdata.get("spread_mean")) if gdata.get("spread_mean") is not None else None),
                spread_std=(self._safe_float(gdata.get("spread_std")) if gdata.get("spread_std") is not None else None),
            )
            mw.groups[gid] = gw
        return mw

    def _load_adaptive_profiles(self) -> None:
        """Load learned user-specific motion profiles from disk."""
        if not self.adaptive_profile_path.exists():
            return

        try:
            with open(self.adaptive_profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[learn] Failed to read adaptive profiles: {e}")
            return

        gestures = payload.get("gestures", {}) if isinstance(payload, dict) else {}
        loaded = 0
        for gesture_key, item in gestures.items():
            if not isinstance(item, dict):
                continue
            metrics = self._deserialize_motion_window(item.get("metrics", {}))
            if not metrics.groups:
                continue
            self._adaptive_profiles[gesture_key] = {
                "samples": max(0, int(item.get("samples", 0))),
                "updated_at": self._safe_float(item.get("updated_at"), 0.0),
                "metrics": metrics,
            }
            loaded += 1

        if loaded:
            print(f"[learn] Loaded {loaded} adaptive gesture profiles")

    def _save_adaptive_profiles(self) -> None:
        """Persist learned profiles to disk."""
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "gestures": {},
        }

        for gesture_key, item in self._adaptive_profiles.items():
            metrics = item.get("metrics")
            if not isinstance(metrics, MotionWindow) or not metrics.groups:
                continue
            payload["gestures"][gesture_key] = {
                "samples": int(item.get("samples", 0)),
                "updated_at": self._safe_float(item.get("updated_at"), 0.0),
                "metrics": self._serialize_motion_window(metrics),
            }

        try:
            self.adaptive_profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.adaptive_profile_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._adaptive_dirty_updates = 0
        except Exception as e:
            print(f"[learn] Failed to save adaptive profiles: {e}")

    def _blend_group_windows(self, base: GroupWindow, other: GroupWindow, other_weight: float) -> GroupWindow:
        """Blend two GroupWindow objects with weighted averages."""
        w = float(np.clip(other_weight, 0.0, 1.0))
        a = 1.0 - w

        def _blend_dict(d1: dict[str, float], d2: dict[str, float]) -> dict[str, float]:
            out: dict[str, float] = {}
            keys = set(d1.keys()) | set(d2.keys())
            for key in keys:
                v1 = d1.get(key)
                v2 = d2.get(key)
                if v1 is None:
                    out[key] = float(v2)
                elif v2 is None:
                    out[key] = float(v1)
                else:
                    out[key] = float(a * v1 + w * v2)
            return out

        dom_dir = None
        if base.dominant_dir or other.dominant_dir:
            bx, by = base.dominant_dir if base.dominant_dir else (0.0, 0.0)
            ox, oy = other.dominant_dir if other.dominant_dir else (0.0, 0.0)
            vx = a * float(bx) + w * float(ox)
            vy = a * float(by) + w * float(oy)
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > 1e-8:
                dom_dir = (vx / mag, vy / mag)

        def _blend_optional(v1: Optional[float], v2: Optional[float], default: Optional[float] = None) -> Optional[float]:
            if v1 is None and v2 is None:
                return default
            if v1 is None:
                return float(v2)
            if v2 is None:
                return float(v1)
            return float(a * v1 + w * v2)

        return GroupWindow(
            group_id=base.group_id,
            present_ratio=float(a * base.present_ratio + w * other.present_ratio),
            angle_mean=_blend_dict(base.angle_mean, other.angle_mean),
            angle_std=_blend_dict(base.angle_std, other.angle_std),
            ratio_mean=_blend_dict(base.ratio_mean, other.ratio_mean),
            ratio_std=_blend_dict(base.ratio_std, other.ratio_std),
            dominant_dir=dom_dir,
            dir_consistency=float(a * base.dir_consistency + w * other.dir_consistency),
            spread_mean=_blend_optional(base.spread_mean, other.spread_mean),
            spread_std=_blend_optional(base.spread_std, other.spread_std),
        )

    def _blend_motion_windows(self, base: MotionWindow, other: MotionWindow, other_weight: float) -> MotionWindow:
        """Blend two MotionWindow objects by group id."""
        result = MotionWindow()
        all_gids = set(base.groups.keys()) | set(other.groups.keys())

        for gid in all_gids:
            base_g = base.groups.get(gid)
            other_g = other.groups.get(gid)
            if base_g and other_g:
                result.groups[gid] = self._blend_group_windows(base_g, other_g, other_weight)
            elif base_g:
                result.groups[gid] = base_g
            elif other_g:
                result.groups[gid] = other_g

        return result

    def _get_scoring_template_metrics(self, gesture_key: str, base_template_mw: MotionWindow) -> tuple[MotionWindow, float, int]:
        """Return template metrics (no adaptive blending - online learning disabled)."""
        return base_template_mw, 0.0, 0

    def _update_adaptive_profile(self, gesture_key: str, live_mw: MotionWindow) -> None:
        """Update per-gesture learned profile via EMA and flush periodically."""
        if not self.online_learning_enabled:
            return
        if gesture_key not in self.action_templates:
            return

        now = time.time()
        profile = self._adaptive_profiles.get(gesture_key)

        if profile and isinstance(profile.get("metrics"), MotionWindow):
            old_metrics: MotionWindow = profile["metrics"]
            new_metrics = self._blend_motion_windows(old_metrics, live_mw, self.adaptive_learning_rate)
            sample_count = int(profile.get("samples", 0)) + 1
        else:
            new_metrics = live_mw
            sample_count = 1

        self._adaptive_profiles[gesture_key] = {
            "samples": sample_count,
            "updated_at": now,
            "metrics": new_metrics,
        }
        self._last_adaptive_update_t = now
        self._adaptive_dirty_updates += 1

        # Persist every few updates to reduce IO overhead.
        if self._adaptive_dirty_updates >= 3:
            self._save_adaptive_profiles()

    def set_online_learning(self, enabled: bool) -> None:
        """Enable or disable adaptive online learning at runtime."""
        self.online_learning_enabled = bool(enabled)
        if not self.online_learning_enabled and self._adaptive_dirty_updates > 0:
            self._save_adaptive_profiles()

    def reload_action_library(self) -> None:
        """Reload action metadata/templates from gesture directory."""
        gesture_dir = str(self.gesture_manager.gesture_dir)
        self.gesture_manager = GestureManager(gesture_dir)
        self.motion_gestures = self.gesture_manager.get_all(category="actions")
        self.action_templates = self._load_action_templates()
        self.has_static_metric_templates = any(
            tpl.get("version") == 2 and not tpl.get("requires_motion", True)
            for tpl in self.action_templates.values()
        )
        self._stable_gesture = None
        self._stable_hits = 0
        print(f"[motion] Reloaded action library: {len(self.action_templates)} templates")

    def save_adaptive_profiles(self) -> None:
        """Public API to force-save learned profiles immediately."""
        self._save_adaptive_profiles()

    def get_adaptive_learning_stats(self) -> dict[str, Any]:
        """Return runtime and persistence stats for adaptive learning."""
        per_gesture = {
            gesture_key: int(item.get("samples", 0))
            for gesture_key, item in self._adaptive_profiles.items()
        }
        return {
            "enabled": bool(self.online_learning_enabled),
            "profile_path": str(self.adaptive_profile_path),
            "profiles": int(len(self._adaptive_profiles)),
            "dirty_updates": int(self._adaptive_dirty_updates),
            "min_samples_for_scoring": int(self.adaptive_min_samples_for_scoring),
            "max_blend_weight": float(self.adaptive_max_blend_weight),
            "update_min_score": float(self.adaptive_update_min_score),
            "update_min_margin": float(self.adaptive_update_min_margin),
            "update_cooldown_sec": float(self.adaptive_update_cooldown_sec),
            "per_gesture_samples": per_gesture,
        }

    def _load_action_templates(self) -> dict[str, dict]:
        """Load all V2 metric gesture templates"""
        templates: dict[str, dict] = {}

        for gesture_key, meta in self.gesture_manager.metadata.items():
            if meta.get("category") != "actions":
                continue

            if gesture_key in self.EXCLUDED_ACTION_KEYS:
                print(f"[motion] Skipping excluded action template: {gesture_key}")
                continue

            file_path = meta.get("file")
            if not file_path:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Validate required_hands location
                if "logic_rules" in data and "required_hands" in data.get("logic_rules", {}):
                    top_level_rh = data.get("required_hands")
                    logic_rh = data["logic_rules"]["required_hands"]
                    if top_level_rh is None:
                        print(f"[ERROR] {gesture_key}: required_hands only in logic_rules (should be top-level)")
                        print(f"  → Will default to inferred value, may cause detection failures!")
                        print(f"  → File: {file_path}")
                    elif top_level_rh != logic_rh:
                        print(f"[WARN] {gesture_key}: required_hands mismatch - top={top_level_rh}, logic={logic_rh}")
                        print(f"  → Using top-level value: {top_level_rh}")

            except Exception as e:
                print(f"[motion] Failed to load template {gesture_key}: {e}")
                continue

            # All templates must be V2 - 3 format
            is_v2 = data.get("version") in (2, 3) or "metrics" in data
            if not is_v2:
                print(f"[motion] Skipping non-V2 gesture: {gesture_key}")
                continue

            tracked = data.get("tracked_groups", ["C1", "C2", "C3", "C4", "C5", "C6"])
            required_hands_raw = data.get("required_hands")
            if required_hands_raw is None:
                # Default inference from tracked hand groups (C2/C4 in current group schema).
                inferred = sum(1 for gid in tracked if gid in {"C2", "C4"})
                required_hands = min(2, max(0, inferred))
            else:
                required_hands = min(2, max(0, int(required_hands_raw)))

            template = {
                "id": data.get("id", gesture_key),
                "name": data.get("name", gesture_key.upper()),
                "version": 2,
                "threshold": data.get("threshold", self.confidence_threshold),
                "tracked_groups": tracked,
                "required_hands": required_hands,
                "hand_presence_min_ratio": float(data.get("hand_presence_min_ratio", 0.45)),
                "hand_presence_expected_factor": float(data.get("hand_presence_expected_factor", 0.55)),
                "template_metrics": extract_template_metrics(data),
                "motion_pattern": data.get("motion_pattern"),
                "spatial_template": data.get("spatial_template"),
                "logic_rules": data.get("logic_rules"),
                "dominant_axis": data.get("dominant_axis"),
                "motion_type": data.get("motion_type"),
                "requires_motion": (
                    bool(data["requires_motion"])
                    if "requires_motion" in data
                    else self._template_requires_motion(data, tracked)
                ),
            }

            templates[gesture_key] = template

        return templates

    def _template_requires_motion(self, data: dict, tracked: list) -> bool:
        """Check if gesture template requires motion or can match static positions"""
        # Extract template metrics
        template_metrics = extract_template_metrics(data)
        if not template_metrics or not tracked:
            return True

        # Check if any group has significant direction consistency
        max_consistency = 0.0
        for gid in tracked:
            gw = template_metrics.groups.get(gid)
            if gw:
                max_consistency = max(max_consistency, gw.dir_consistency)

        # If max_consistency < 0.30, gesture is mostly static
        return max_consistency >= 0.30

    @staticmethod
    def _mirror_group_id(gid: str) -> str:
        mirror = {"C1": "C3", "C2": "C4", "C3": "C1", "C4": "C2"}
        return mirror.get(gid, gid)

    @staticmethod
    def _mirror_angle_deg(theta: float) -> float:
        # Horizontal mirror around vertical axis: atan2(y, x) -> atan2(y, -x)
        return (180.0 - float(theta)) % 360.0

    def _build_mirrored_template(self, template_metrics):
        mirrored = MotionWindow()
        for gid, gw in template_metrics.groups.items():
            mirrored_gid = self._mirror_group_id(gid)

            mirrored_angle_mean = {}
            for k, v in gw.angle_mean.items():
                if "orient" in k or "roll" in k:
                    mirrored_angle_mean[k] = self._mirror_angle_deg(v)
                else:
                    mirrored_angle_mean[k] = v

            mirrored_dom_dir = None
            if gw.dominant_dir is not None:
                mirrored_dom_dir = (-gw.dominant_dir[0], gw.dominant_dir[1])

            mirrored.groups[mirrored_gid] = GroupWindow(
                group_id=mirrored_gid,
                present_ratio=gw.present_ratio,
                angle_mean=mirrored_angle_mean,
                angle_std=dict(gw.angle_std),
                ratio_mean=dict(gw.ratio_mean),
                ratio_std=dict(gw.ratio_std),
                dominant_dir=mirrored_dom_dir,
                dir_consistency=gw.dir_consistency,
                spread_mean=gw.spread_mean,
                spread_std=gw.spread_std,
            )
        return mirrored

    # Static gestures only need to match the most recent few frames
    STATIC_EVAL_WINDOW = 8
    MOTION_EVAL_WINDOW = 12

    def _compute_bilateral_wrist_motion(self, eval_buffer: list[dict[str, Any]]) -> dict[str, float]:
        """Compute bilateral wrist distance dynamics from group state frames.

        Returns normalized features used to separate clap vs wave:
        - min_dist: minimum L/R wrist distance (0..sqrt(2), lower = closer)
        - max_dist: maximum L/R wrist distance
        - start_dist: mean distance in earliest third of window
        - end_dist: mean distance in latest third of window
        - closing_delta: early_mean - late_mean (positive = moving together)
        """
        dists: list[float] = []

        for frame in eval_buffer:
            c2 = frame.get("C2")
            c4 = frame.get("C4")
            if not c2 or not c4:
                continue
            if not c2.present or not c4.present:
                continue
            if not c2.anchor_pos or not c4.anchor_pos:
                continue
            dx = float(c2.anchor_pos[0]) - float(c4.anchor_pos[0])
            dy = float(c2.anchor_pos[1]) - float(c4.anchor_pos[1])
            dists.append(math.sqrt(dx * dx + dy * dy))

        if len(dists) < 3:
            return {
                "samples": float(len(dists)),
                "min_dist": 1.0,
                "max_dist": 1.0,
                "start_dist": 1.0,
                "end_dist": 1.0,
                "closing_delta": 0.0,
                "contact_ratio": 0.0,
                "open_to_contact_span": 0.0,
            }

        seg = max(1, len(dists) // 3)
        early = dists[:seg]
        late = dists[-seg:]
        min_dist = float(min(dists))
        max_dist = float(max(dists))
        start_dist = float(sum(early) / len(early))
        end_dist = float(sum(late) / len(late))

        # Portion of frames where wrists are in "contact/near-contact" zone.
        contact_thr = 0.13
        contact_ratio = float(sum(1 for d in dists if d <= contact_thr) / len(dists))

        return {
            "samples": float(len(dists)),
            "min_dist": min_dist,
            "max_dist": max_dist,
            "start_dist": start_dist,
            "end_dist": end_dist,
            "closing_delta": float(start_dist - end_dist),
            "contact_ratio": contact_ratio,
            "open_to_contact_span": float(max_dist - min_dist),
        }

    def clear_buffers(self) -> None:
        """
        Xóa toàn bộ Cửa sổ hiện tại sau khi đã chốt được một hành động ký hiệu.
        Điều này đảm bảo cho chuỗi hành động kế tiếp không bị dính những
        hình ảnh thu tay về của hành động kết thúc trước đó.
        """
        self.hand_positions.clear()
        self.hand_landmarks_buffer.clear()
        self.pose_landmarks_buffer.clear()
        self.group_state_buffer.clear()
        self.snapshot_buffer.clear()
        self._missing_frame_count = 0
        self._stable_gesture = None
        self._stable_hits = 0
        self.group_tracker.reset()
        print("[MotionMatcher] Segment consumed. Buffer cleared for next gesture token in stream.")

    def get_peak_records(self) -> dict:
        """Trả về dict các gesture đã từng đạt ≥ 80% trong session hiện tại."""
        return dict(self._peak_records)

    def reset_peak_records(self) -> None:
        """Xóa peak records (gọi khi bắt đầu session mới)."""
        self._peak_records.clear()

    def _find_gesture_boundaries_by_velocity(self) -> list:
        """
        Thuật toán Dynamic Alignment:
        Trượt trên dòng stream hiện tại để dò tìm 2 lần vận tốc thả tay bằng 0,
        từ đó trích xuất ra một đoạn Window độc lập chứa trọn vẹn hành động
        ngăn nguy cơ cắt phạm chuỗi dài.
        """
        buffer_list = list(self.group_state_buffer)
        
        # Nếu chưa đủ dài chỉ đánh giá cái hiện tại
        if len(buffer_list) < 5:
            return buffer_list
            
        # Ở đây hệ thống nội suy khoảng cách và tốc độ
        # Tạm thời mở tung toàn bộ cửa sổ lưu trữ cho phép phân tích nguyên vẹn 100% dòng Frame
        return buffer_list

    def _score_motion_pattern(
        self,
        pattern: dict[str, Any],
        wrist_motion: dict[str, float],
        live_mw: MotionWindow,
    ) -> float:
        """Score motion pattern constraints defined in template JSON (0..1)."""
        min_dist = float(wrist_motion.get("min_dist", 1.0))
        max_dist = float(wrist_motion.get("max_dist", 1.0))
        start_dist = float(wrist_motion.get("start_dist", 1.0))
        end_dist = float(wrist_motion.get("end_dist", 1.0))
        closing_delta = float(wrist_motion.get("closing_delta", 0.0))
        open_to_contact_span = float(wrist_motion.get("open_to_contact_span", 0.0))
        contact_ratio = float(wrist_motion.get("contact_ratio", 0.0))

        # Chest-zone proxy from arm groups (wrist_height).
        chest_vals = []
        for gid in ("C1", "C3"):
            gw = live_mw.groups.get(gid)
            if gw and "wrist_height" in gw.ratio_mean:
                chest_vals.append(float(gw.ratio_mean["wrist_height"]))
        if chest_vals:
            chest_h = float(sum(chest_vals) / len(chest_vals))
        else:
            chest_h = 0.10

        def _score_ge(value: float, threshold: float, scale: float) -> float:
            return float(np.clip((value - threshold) / max(scale, 1e-6), 0.0, 1.0))

        def _score_le(value: float, threshold: float, scale: float) -> float:
            return float(np.clip((threshold - value) / max(scale, 1e-6), 0.0, 1.0))

        weights = pattern.get("weights", {}) if isinstance(pattern.get("weights"), dict) else {}
        total = 0.0
        wsum = 0.0

        def _add(score: float, key: str, default_w: float):
            nonlocal total, wsum
            w = float(weights.get(key, default_w))
            if w <= 0.0:
                return
            total += score * w
            wsum += w

        if "start_dist_min" in pattern:
            thr = float(pattern.get("start_dist_min", 0.0))
            scl = float(pattern.get("start_dist_scale", 0.18))
            _add(_score_ge(start_dist, thr, scl), "start_dist", 0.14)
        if "max_dist_min" in pattern:
            thr = float(pattern.get("max_dist_min", 0.0))
            scl = float(pattern.get("max_dist_scale", 0.20))
            _add(_score_ge(max_dist, thr, scl), "max_dist", 0.08)

        if "min_dist_max" in pattern:
            thr = float(pattern.get("min_dist_max", 1.0))
            scl = float(pattern.get("min_dist_scale", 0.13))
            _add(_score_le(min_dist, thr, scl), "min_dist", 0.18)
        if "min_dist_min" in pattern:
            thr = float(pattern.get("min_dist_min", 0.0))
            scl = float(pattern.get("min_dist_min_scale", 0.10))
            _add(_score_ge(min_dist, thr, scl), "min_dist_min", 0.18)

        if "end_dist_max" in pattern:
            thr = float(pattern.get("end_dist_max", 1.0))
            scl = float(pattern.get("end_dist_scale", 0.16))
            _add(_score_le(end_dist, thr, scl), "end_dist", 0.12)
        if "end_dist_min" in pattern:
            thr = float(pattern.get("end_dist_min", 0.0))
            scl = float(pattern.get("end_dist_min_scale", 0.10))
            _add(_score_ge(end_dist, thr, scl), "end_dist_min", 0.12)

        if "closing_delta_min" in pattern:
            thr = float(pattern.get("closing_delta_min", 0.0))
            scl = float(pattern.get("closing_delta_scale", 0.12))
            _add(_score_ge(closing_delta, thr, scl), "closing_delta", 0.16)
        if "closing_delta_max" in pattern:
            thr = float(pattern.get("closing_delta_max", 1.0))
            scl = float(pattern.get("closing_delta_max_scale", 0.08))
            _add(_score_le(closing_delta, thr, scl), "closing_delta_max", 0.16)

        if "span_min" in pattern:
            thr = float(pattern.get("span_min", 0.0))
            scl = float(pattern.get("span_scale", 0.20))
            _add(_score_ge(open_to_contact_span, thr, scl), "span", 0.10)
        if "contact_ratio_min" in pattern:
            thr = float(pattern.get("contact_ratio_min", 0.0))
            scl = float(pattern.get("contact_ratio_scale", 0.40))
            _add(_score_ge(contact_ratio, thr, scl), "contact_ratio", 0.08)

        if "chest_height_target" in pattern:
            target = float(pattern.get("chest_height_target", 0.10))
            tol = float(pattern.get("chest_height_tol", 0.25))
            chest_score = float(np.clip(1.0 - abs(chest_h - target) / max(tol, 1e-6), 0.0, 1.0))
            _add(chest_score, "chest_height", 0.08)

        raw = (total / wsum) if wsum > 0 else 0.0

        # Optional soft-gate rules from JSON.
        gate_rules = pattern.get("gate", [])
        if isinstance(gate_rules, list) and gate_rules:
            gate_ok = True
            for key in gate_rules:
                if key == "start_dist_min" and not ("start_dist_min" in pattern and start_dist >= float(pattern["start_dist_min"])):
                    gate_ok = False
                elif key == "min_dist_max" and not ("min_dist_max" in pattern and min_dist <= float(pattern["min_dist_max"])):
                    gate_ok = False
                elif key == "min_dist_min" and not ("min_dist_min" in pattern and min_dist >= float(pattern["min_dist_min"])):
                    gate_ok = False
                elif key == "end_dist_max" and not ("end_dist_max" in pattern and end_dist <= float(pattern["end_dist_max"])):
                    gate_ok = False
                elif key == "end_dist_min" and not ("end_dist_min" in pattern and end_dist >= float(pattern["end_dist_min"])):
                    gate_ok = False
                elif key == "closing_delta_min" and not ("closing_delta_min" in pattern and closing_delta >= float(pattern["closing_delta_min"])):
                    gate_ok = False
                elif key == "closing_delta_max" and not ("closing_delta_max" in pattern and closing_delta <= float(pattern["closing_delta_max"])):
                    gate_ok = False
                elif key == "span_min" and not ("span_min" in pattern and open_to_contact_span >= float(pattern["span_min"])):
                    gate_ok = False
                elif key == "contact_ratio_min" and not ("contact_ratio_min" in pattern and contact_ratio >= float(pattern["contact_ratio_min"])):
                    gate_ok = False
            if not gate_ok:
                fail_factor = self.gate_fail_factor_override
                if fail_factor is None:
                    fail_factor = float(pattern.get("gate_fail_factor", 0.35))
                raw *= float(np.clip(fail_factor, 0.0, 1.0))

        return float(np.clip(raw, 0.0, 1.0))

    def _score_spatial_template(self, spatial_dict: dict, eval_buffer: list, debug_dict: dict) -> float:
        """
        Subsequence DTW matching cho từng cluster.

        Thay vì resample live path về cùng độ dài template rồi so sánh điểm-với-điểm,
        dùng Subsequence DTW để tìm đoạn trong live_path khớp tốt nhất với template.

        Ưu điểm:
        - Bỏ qua frames nhiễu ở đầu/cuối buffer (tay chưa sẵn, tay chạm nhau)
        - Không bị ảnh hưởng bởi tốc độ thực hiện gesture (nhanh/chậm)
        - Variance có thể giữ chặt hơn → phân biệt gestures tốt hơn
        """
        if not eval_buffer or not spatial_dict:
            return 1.0

        group_scores = []

        for tg, t_data in spatial_dict.items():
            if tg == "weight":
                continue

            anchor      = t_data.get("anchor", "C5")
            dtw_matrix  = t_data.get("dtw_matrix", [])
            weight      = float(t_data.get("weight", 1.0))
            variance    = float(t_data.get("variance", 0.15))

            if not dtw_matrix:
                continue

            # ── Build live path (relative to anchor) ──────────────────────
            live_path = []
            for st in eval_buffer:
                if tg not in st or not st[tg].present or not st[tg].anchor_pos:
                    continue
                a_pos = None
                if anchor in st and st[anchor].present and st[anchor].anchor_pos:
                    a_pos = st[anchor].anchor_pos
                elif "C5" in st and st["C5"].present and st["C5"].anchor_pos:
                    a_pos = st["C5"].anchor_pos  # fallback torso
                if a_pos:
                    live_path.append([
                        st[tg].anchor_pos[0] - a_pos[0],
                        st[tg].anchor_pos[1] - a_pos[1],
                    ])

            if len(live_path) < 3:
                group_scores.append(0.0)
                debug_dict[f"spatial_{tg}_dist"] = 999.0
                debug_dict[f"spatial_{tg}_scr"]  = 0.0
                continue

            # ── Subsequence DTW ───────────────────────────────────────────
            # Query = template dtw_matrix (N keyframes)
            # Sequence = live_path (M frames, M >= N)
            # Free start: sub-sequence can begin anywhere in live_path
            # Goal: find contiguous sub-sequence of live_path minimizing DTW dist to query

            query = dtw_matrix
            seq   = live_path
            n     = len(query)
            m     = len(seq)

            INF  = float("inf")
            # prev[j] = DTW cost ending at seq[j-1] after processing query[0..i-1]
            prev = [0.0] + [INF] * m  # free start: cost(0,j) = 0 for all j

            for i in range(1, n + 1):
                curr = [INF] * (m + 1)
                qx, qy = query[i - 1]
                for j in range(1, m + 1):
                    sx, sy = seq[j - 1]
                    step_cost = math.sqrt((qx - sx) ** 2 + (qy - sy) ** 2)
                    curr[j] = step_cost + min(
                        prev[j],      # query advances, seq stays (insertion)
                        curr[j - 1],  # seq advances, query stays (deletion)
                        prev[j - 1],  # both advance (match)
                    )
                prev = curr

            # Best ending position in seq (minimum cost over all end positions)
            best_cost = min(prev[1:])
            avg_dist  = best_cost / max(1, n)

            # ── Convert dist → score ──────────────────────────────────────
            margin = variance * 1.5
            if avg_dist <= variance:
                sub_score = 1.0
            elif avg_dist >= variance + margin:
                sub_score = 0.0
            else:
                sub_score = 1.0 - (avg_dist - variance) / margin

            group_scores.append(sub_score * weight)
            debug_dict[f"spatial_{tg}_dist"] = round(avg_dist, 3)
            debug_dict[f"spatial_{tg}_scr"]  = round(sub_score, 2)

        if not group_scores:
            return 1.0
        return sum(group_scores) / len(group_scores)

    def _score_logic_rules(self, rules: dict, eval_buffer: list, debug_dict: dict) -> float:
        if "sequence" in rules:
            sequence = rules["sequence"]
            total_frames = len(eval_buffer)
            if total_frames == 0:
                return 0.0
            
            seq_score = 0.0
            start_idx = 0
            for idx, phase in enumerate(sequence):
                time_frac = float(phase.get("time_fraction", 1.0 / len(sequence)))
                end_idx = start_idx + int(time_frac * total_frames)
                if idx == len(sequence) - 1:
                    end_idx = total_frames
                
                phase_buffer = eval_buffer[start_idx:end_idx]
                if not phase_buffer:
                    continue
                
                phase_dbg = {}
                phase_score = self._score_logic_rules(phase, phase_buffer, phase_dbg)
                debug_dict[f"phase_{idx}_{phase.get('phase', 'unnamed')}"] = phase_dbg
                seq_score += phase_score * time_frac
                start_idx = end_idx
            
            return seq_score

        score = 1.0 # start at 1.0, penalties apply
        
        # 1. Hand Pose rule
        if "hand_pose" in rules:
            hp = rules["hand_pose"]
            
            # Check if using the new format separating left and right hand
            if "left_hand" in hp or "right_hand" in hp:
                left_pose = hp.get("left_hand", {}).get("state")
                left_weight = float(hp.get("left_hand", {}).get("weight", 0.0))
                right_pose = hp.get("right_hand", {}).get("state")
                right_weight = float(hp.get("right_hand", {}).get("weight", 0.0))
                
                left_matches, right_matches, total_frames = 0, 0, 0
                for st in eval_buffer:
                    has_left = "C2" in st and st["C2"].present and st["C2"].hand_pose
                    has_right = "C4" in st and st["C4"].present and st["C4"].hand_pose
                    
                    if has_left or has_right:
                        total_frames += 1
                        if has_left and left_pose and left_pose in st["C2"].hand_pose:
                            left_matches += 1
                        if has_right and right_pose and right_pose in st["C4"].hand_pose:
                            right_matches += 1

                if total_frames > 0:
                    left_ratio = left_matches / total_frames if left_pose else 1.0
                    right_ratio = right_matches / total_frames if right_pose else 1.0
                    
                    if left_pose:
                        score -= (1.0 - left_ratio) * left_weight
                    if right_pose:
                        score -= (1.0 - right_ratio) * right_weight
                    debug_dict["left_pose_ratio"] = left_ratio
                    debug_dict["right_pose_ratio"] = right_ratio
            else:
                expected_pose = hp.get("state")
                weight = float(hp.get("weight", 0.2))
                
                # Check dominant hand sequence
                pose_matches, total_frames = 0, 0
                for st in eval_buffer:
                    for tg in ("C2", "C4"):
                        if tg in st and st[tg].present and st[tg].hand_pose:
                            total_frames += 1
                            if expected_pose in st[tg].hand_pose: pose_matches += 1
                pose_ratio = pose_matches / max(1, total_frames) if total_frames > 0 else 0.0
                debug_dict["hand_pose_ratio"] = pose_ratio
                score -= (1.0 - pose_ratio) * weight
        
        # 1.2 Dynamic Spatial Relations 
        if "dynamic_spatial_relations" in rules:
            relations = rules["dynamic_spatial_relations"]
            for rel in relations:
                # "right_hand_to_left_hand" means evaluating Right Hand relative to Left Hand
                target = rel.get("target", "right_hand_to_left_hand")
                expected = rel.get("expected_relation")
                weight = float(rel.get("weight", 0.3))
                max_dist = float(rel.get("max_distance", 0.4))
                
                match_count, total_frames, violation_count = 0, 0, 0
                for st in eval_buffer:
                    if target in ["hands", "right_hand_to_left_hand"] and "C2" in st and st["C2"].present and "C4" in st and st["C4"].present:
                        total_frames += 1
                        left_pos = st["C2"].anchor_pos
                        right_pos = st["C4"].anchor_pos
                        
                        if left_pos and right_pos:
                            rel_x = right_pos[0] - left_pos[0]
                            rel_y = right_pos[1] - left_pos[1]
                            dist = math.hypot(rel_x, rel_y)
                            
                            is_match = False
                            is_severe_violation = False
                            
                            if expected == "contact":
                                if dist < 0.15: is_match = True
                                elif dist > 0.4: is_severe_violation = True # xa quá
                            elif expected == "above": # Right hand above left hand -> y is smaller
                                if rel_y < -0.05 and dist < max_dist: is_match = True
                                elif rel_y > 0.1: is_severe_violation = True  # bị ngược (dưới thay vì trên)
                            elif expected == "below":
                                if rel_y > 0.05 and dist < max_dist: is_match = True
                                elif rel_y < -0.1: is_severe_violation = True 
                            elif expected == "left_of": # Right hand is to the left of Left hand -> x is smaller
                                if rel_x < -0.05 and dist < max_dist: is_match = True
                                elif rel_x > 0.1: is_severe_violation = True
                            elif expected == "right_of": # Right hand is to the right of Left hand
                                if rel_x > 0.05 and dist < max_dist: is_match = True
                                elif rel_x < -0.1: is_severe_violation = True
                                
                            # Cũ hỗ trợ tương thích ngược:
                            elif expected == "left_hand_above" and rel_y > 0.05 and dist < max_dist: is_match = True
                            elif expected == "right_hand_above" and rel_y < -0.05 and dist < max_dist: is_match = True
                            
                            if is_match:
                                match_count += 1
                            if is_severe_violation:
                                violation_count += 1
                
                if total_frames > 0:
                    ratio = match_count / total_frames
                    violation_ratio = violation_count / total_frames
                    debug_dict[f"spatial_{expected}_ratio"] = ratio
                    debug_dict[f"spatial_{expected}_violation"] = violation_ratio
                    
                    # Negative Threshold: Vi phạm trầm trọng dẫn cấu trúc tay (2 tay sai vị trí nhau) => ngắt mạnh
                    if violation_ratio > 0.5:
                        score -= weight * 2.0  # Phạt nặng ngắt score
                    else:
                        score -= (1.0 - ratio) * weight
        
        # 1.5. Require Hand Pose
        if "require_hand_pose" in rules:
            req_pose = rules["require_hand_pose"]
            pose_name = req_pose.get("pose", "")
            w = float(req_pose.get("weight", 0.5))
            
            # Count frames where this pose is seen
            # Usually only evaluates when fingers/hands are detected
            pose_matches = 0
            total_frames = 0
            for st in eval_buffer:
                for tg in ("C2", "C4"):
                    if tg in st and st[tg].present and st[tg].hand_pose:
                        total_frames += 1
                        if pose_name in st[tg].hand_pose:
                            pose_matches += 1
            
            req_ratio = pose_matches / max(1, total_frames) if total_frames > 0 else 0.0
            debug_dict["req_pose_ratio"] = req_ratio
            
            # Penalize heavily if the ratio is too low
            score -= (1.0 - req_ratio) * w
# 2. Arm Kinematics (Oscillation / Angles) rules
        if "arm_kinematics" in rules:
            kin = rules["arm_kinematics"]
            r_min, r_max = kin.get("angle_range_deg", [0, 360])
            req_swings = kin.get("repetitive_swings", 0)
            weight = float(kin.get("weight", 0.8))
            
            c1_angles, c3_angles = [], []
            for st in eval_buffer:
                if "C1" in st and st["C1"].present and getattr(st["C1"], "forearm_body_angle", None) is not None:
                    c1_angles.append(st["C1"].forearm_body_angle)
                if "C3" in st and st["C3"].present and getattr(st["C3"], "forearm_body_angle", None) is not None:
                    c3_angles.append(st["C3"].forearm_body_angle)
            
            def count_swings(angles: list) -> int:
                if len(angles) < 3: return 0
                swings = 0
                smoothed = [sum(angles[i:i+3])/3 for i in range(len(angles)-2)]
                dir = None
                for i in range(1, len(smoothed)):
                    diff = smoothed[i] - smoothed[i-1]
                    if abs(diff) > 180: diff = diff - math.copysign(360, diff)
                    if abs(diff) > 5.0: # avoid jitter
                        new_dir = 1 if diff > 0 else -1
                        if dir is not None and new_dir != dir:
                            swings += 1
                        dir = new_dir
                return swings

            def is_in_range(angles: list) -> float:
                if not angles: return 0.0
                in_r_count = sum(1 for a in angles if r_min <= a <= r_max)
                return in_r_count / len(angles)

            s1 = count_swings(c1_angles)
            s3 = count_swings(c3_angles)
            req_hands = rules.get("required_hands", 1)  # allow passing required hands via rules
            
            if req_hands == 2:
                # Both hands must swing
                eff_swings = min(s1, s3)
                best_arm_angles = c1_angles if s1 <= s3 else c3_angles # Use the worse arm to test range constraint
            else:
                eff_swings = max(s1, s3)
                best_arm_angles = c1_angles if s1 >= s3 else c3_angles
                
            range_ratio = is_in_range(best_arm_angles)
            
            debug_dict["swings"] = eff_swings
            debug_dict["range_ratio"] = range_ratio
            debug_dict["best_arm_angles_count"] = len(best_arm_angles)

            swing_score = min(eff_swings / max(1, req_swings), 1.0) if req_swings > 0 else 1.0
            kin_score = (swing_score * 0.7) + (range_ratio * 0.3)
            score -= (1.0 - kin_score) * weight

        # 2.5 Wrist Kinematics (Nodding / Palm angles)
        if "wrist_kinematics" in rules:
            w_kin = rules["wrist_kinematics"]
            req_swings = w_kin.get("repetitive_swings", 0)
            weight = float(w_kin.get("weight", 0.6))
            
            c2_angles, c4_angles = [], []
            for st in eval_buffer:
                if "C2" in st and st["C2"].present and "palm_orient" in st["C2"].angles:
                    c2_angles.append(st["C2"].angles["palm_orient"])
                if "C4" in st and st["C4"].present and "palm_orient" in st["C4"].angles:
                    c4_angles.append(st["C4"].angles["palm_orient"])
            
            def count_palm_swings(angles: list) -> int:
                if len(angles) < 3: return 0
                swings = 0
                smoothed = [sum(angles[i:i+3])/3 for i in range(len(angles)-2)]
                dir = None
                for i in range(1, len(smoothed)):
                    diff = smoothed[i] - smoothed[i-1]
                    if abs(diff) > 180: diff = diff - math.copysign(360, diff)
                    if abs(diff) > 2.0: # palm jitter threshold
                        new_dir = 1 if diff > 0 else -1
                        if dir is not None and new_dir != dir:
                            swings += 1
                        dir = new_dir
                return swings

            w_s1 = count_palm_swings(c2_angles)
            w_s3 = count_palm_swings(c4_angles)
            eff_swings = max(w_s1, w_s3)
            swing_score = min(eff_swings / max(1, req_swings), 1.0) if req_swings > 0 else 1.0
            
            debug_dict["wrist_swings"] = eff_swings
            score -= (1.0 - swing_score) * weight

        # 3. Movement / Trajectory rules
        if "movement" in rules:
            mv = rules["movement"]
            axis = mv.get("axis", "vertical")
            req_swings = mv.get("repetitive_swings", 0)
            direction = mv.get("direction", None) # "up", "down", "left", "right"
            weight = float(mv.get("weight", 0.8))
            
            c2_pos, c4_pos = [], []
            for st in eval_buffer:
                if "C2" in st and getattr(st["C2"], "anchor_pos", None):
                    c2_pos.append(st["C2"].anchor_pos)
                if "C4" in st and getattr(st["C4"], "anchor_pos", None):
                    c4_pos.append(st["C4"].anchor_pos)
            
            # Use Y for vertical, X for horizontal
            idx_axis = 1 if axis == "vertical" else 0
            c2_vals = [p[idx_axis] for p in c2_pos]
            c4_vals = [p[idx_axis] for p in c4_pos]
            
            def count_trajectory_swings(vals):
                if len(vals) < 3: return 0
                swings = 0
                smoothed = [sum(vals[i:i+3])/3 for i in range(len(vals)-2)]
                dir = None
                for i in range(1, len(smoothed)):
                    diff = smoothed[i] - smoothed[i-1]
                    if abs(diff) > 0.02: # threshold for normalized screen pos
                        new_dir = 1 if diff > 0 else -1
                        if dir is not None and new_dir != dir:
                            swings += 1
                        dir = new_dir
                return swings
                
            eff_swings = max(count_trajectory_swings(c2_vals), count_trajectory_swings(c4_vals))
            swing_score = min(eff_swings / max(1, req_swings), 1.0) if req_swings > 0 else 1.0
            
            # Check uni-directional
            dir_score = 1.0
            if direction:
                vals = c2_vals if len(c2_vals) > len(c4_vals) else c4_vals
                if len(vals) > 0:
                    delta = vals[-1] - vals[0]
                    if direction == "up" and delta > -0.05: dir_score = 0.0
                    elif direction == "down" and delta < 0.05: dir_score = 0.0
                    elif direction == "left" and delta > -0.05: dir_score = 0.0
                    elif direction == "right" and delta < 0.05: dir_score = 0.0

            mv_score = swing_score * dir_score
            score -= (1.0 - mv_score) * weight

        # 4. Location rules (e.g. hand near chest / face)
        if "location" in rules:
            loc = rules["location"]
            target = loc.get("target") # "chest", "face", "side_face", "side_chest", "face_left", "face_right", "chest_left", "chest_right"
            weight = float(loc.get("weight", 0.6))
            dist_threshold = loc.get("distance_threshold", 0.15) # default max distance 15% of screen

            # Khắc phục Dual-Check / Region Overlap bằng Soft-Margin: 
            # Mở rộng bán kính linh hoạt (2.5x) thay vì giới hạn tuyệt đối để không trượt logic ngực/bụng/hông bị chồng chéo
            soft_margin_multiplier = 2.5 
            near_score_sum, valid_target_frames = 0.0, 0
            for st in eval_buffer:
                for tg in ("C2", "C4"):
                    if tg in st and st[tg].present and st[tg].anchor_pos:
                        hand_pos = st[tg].anchor_pos
                        target_pos = None

                        if target.startswith("face") and "C6" in st and st["C6"].present and st["C6"].anchor_pos:
                            base_x, base_y = st["C6"].anchor_pos
                            if target == "side_face": base_x += 0.20 if hand_pos[0] > base_x else -0.20
                            elif target == "face_left": base_x += 0.20
                            elif target == "face_right": base_x -= 0.20
                            target_pos = (base_x, base_y)

                        elif ("chest" in target) and "C5" in st and st["C5"].present and st["C5"].anchor_pos:
                            base_x, base_y = st["C5"].anchor_pos
                            if target == "side_chest": base_x += 0.20 if hand_pos[0] > base_x else -0.20
                            elif target == "chest_left": base_x += 0.20
                            elif target == "chest_right": base_x -= 0.20
                            target_pos = (base_x, base_y)

                        if target_pos:
                            valid_target_frames += 1
                            dx = hand_pos[0] - target_pos[0]
                            dy = hand_pos[1] - target_pos[1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            
                            # Soft-margin distance evaluation
                            if dist <= dist_threshold:
                                near_score_sum += 1.0
                            elif dist <= dist_threshold * soft_margin_multiplier:
                                # Falloff tuyến tính từ 1.0 xuống 0.0 theo khoảng cách tràn
                                falloff = 1.0 - (dist - dist_threshold) / (dist_threshold * (soft_margin_multiplier - 1.0))
                                near_score_sum += falloff

            if valid_target_frames > 0:
                loc_ratio = near_score_sum / valid_target_frames
                debug_dict["location_ratio"] = loc_ratio
                score -= (1.0 - loc_ratio) * weight
            else:
                debug_dict["location_ratio"] = "N/A (target missing)"
                # Tiny penalty for being completely unable to verify location, but not full weight
                score -= 0.05
        else:
            # Penalize actions that DO NOT require location if hand IS near a specific distinct location 
            # (To prevent a generic hand-swipe from hijacking a specialized hand-to-face gesture)
            near_distinct_frames, total_frames = 0, 0
            for st in eval_buffer:
                for tg in ("C2", "C4"):
                    if tg in st and st[tg].present and st[tg].anchor_pos:
                        total_frames += 1
                        hand_pos = st[tg].anchor_pos
                        # Consider all sensitive bodily areas as distinct targets
                        targets = []
                        if "C6" in st and st["C6"].present and st["C6"].anchor_pos: 
                            hx, hy = st["C6"].anchor_pos
                            targets.extend([(hx, hy), (hx+0.2, hy), (hx-0.2, hy)]) # face center + 2 sides
                        if "C5" in st and st["C5"].present and st["C5"].anchor_pos: 
                            cx, cy = st["C5"].anchor_pos
                            targets.extend([(cx, cy), (cx+0.2, cy), (cx-0.2, cy)]) # chest center + 2 sides
                        
                        for target_pos in targets:
                            if target_pos:
                                dx = hand_pos[0] - target_pos[0]
                                dy = hand_pos[1] - target_pos[1]
                                dist = math.sqrt(dx*dx + dy*dy)
                                if dist < 0.15: # if suspiciously close to a vital area
                                    near_distinct_frames += 1
                                    break # count frame once
            
            # If the json didn't specify location, but the user is doing the gesture right in their face 
            # (like drinking), we penalize this generic rule to avoid theft.
            loc_ratio = near_distinct_frames / max(1, total_frames) if total_frames > 0 else 0.0
            if loc_ratio > 0.3:
                # Deduct heavily if action doesn't declare a location but happens right at the face/chest.
                score -= loc_ratio * 0.5
                debug_dict["location_hijack_penalty"] = loc_ratio * 0.5 

        return max(0.0, score)

    def _detect_metric_gesture(self, gesture_key: str, template: dict) -> Dict[str, Any]:
        """Detect gesture using V2 cluster-based metrics"""
        requires_motion = template.get("requires_motion", True)

        # With sliding-window peak matching, we need minimum 3 frames for meaningful stats
        if requires_motion:
            required_frames = 3  # Need 3 frames for sliding window approach
            eval_buffer = self._find_gesture_boundaries_by_velocity()
        else:
            required_frames = 3  # Need 3 frames for static gesture window
            eval_buffer = list(self.group_state_buffer)[-self.STATIC_EVAL_WINDOW:]

        if len(eval_buffer) < required_frames:
            if self.debug_log_rejections:
                print(f"[REJECT] {gesture_key}: Not enough frames")
                print(f"  → Required: {required_frames}, Current: {len(eval_buffer)}")
                print(f"  → Buffer size: {len(self.group_state_buffer)}")
            return {
                "detected": False,
                "gesture": gesture_key,
                "gesture_name": "Unknown",
                "score": 0.0,
                "debug": {
                    "reason": "Not enough frames",
                    "required_frames": required_frames,
                    "current_frames": len(eval_buffer),
                },
            }

        # ══════════════════════════════════════════════════════════════════
        # PER-FRAME PEAK SCORING
        # Requirement: Check EACH frame individually to find peak similarity
        # ══════════════════════════════════════════════════════════════════
        tracked = template.get("tracked_groups", ["C1", "C2", "C3", "C4", "C5", "C6"])
        mirrored_tracked = [self._mirror_group_id(g) for g in tracked]
        active_groups = sorted(set(tracked + mirrored_tracked))

        # Get template metrics (needed for per-frame comparison)
        base_template_mw = template["template_metrics"]
        template_mw, adaptive_weight, adaptive_samples = self._get_scoring_template_metrics(gesture_key, base_template_mw)

        # Build mirrored template once
        mirrored_template = self._build_mirrored_template(template_mw)

        # Compute per-WINDOW scores to find peak (sliding window approach)
        # Use 3-frame windows instead of single frames to get meaningful statistics
        peak_score = 0.0
        peak_frame_idx = -1
        frame_scores = []

        PEAK_WINDOW_SIZE = 3  # Minimum window size for meaningful angle/ratio std

        if len(eval_buffer) >= PEAK_WINDOW_SIZE:
            # Sliding window approach: score overlapping 3-frame windows
            for start_idx in range(len(eval_buffer) - PEAK_WINDOW_SIZE + 1):
                window = eval_buffer[start_idx:start_idx + PEAK_WINDOW_SIZE]
                window_mw = aggregate_window(window, active_groups=active_groups)

                # Apply motion boost if enabled
                if self.motion_boost_enabled:
                    window_mw = self._apply_motion_boost(window_mw)

                # Score this window against template
                window_score_primary = motion_window_similarity(window_mw, template_mw, tracked)
                window_score_mirrored = motion_window_similarity(window_mw, mirrored_template, mirrored_tracked)
                window_score = max(window_score_primary, window_score_mirrored)

                frame_scores.append(window_score)

                if window_score > peak_score:
                    peak_score = window_score
                    peak_frame_idx = start_idx + (PEAK_WINDOW_SIZE // 2)  # Center of window
        elif len(eval_buffer) > 0:
            # Fallback for very short buffers (1-2 frames): use all available frames
            window_mw = aggregate_window(eval_buffer, active_groups=active_groups)
            if self.motion_boost_enabled:
                window_mw = self._apply_motion_boost(window_mw)

            window_score_primary = motion_window_similarity(window_mw, template_mw, tracked)
            window_score_mirrored = motion_window_similarity(window_mw, mirrored_template, mirrored_tracked)
            peak_score = max(window_score_primary, window_score_mirrored)
            peak_frame_idx = 0

        # ══════════════════════════════════════════════════════════════════
        # AGGREGATE SCORING (existing - for ranking and consistency)
        # ══════════════════════════════════════════════════════════════════
        live_mw = aggregate_window(eval_buffer, active_groups=active_groups)

        # Apply motion boost normalization if enabled (normalizes by body scale and motion cycle)
        if self.motion_boost_enabled:
            live_mw = self._apply_motion_boost(live_mw)

        # Reject early when required groups are mostly absent in both primary and mirrored views.
        # This avoids scoring two-hand gestures highly when only one hand is visible.
        def _presence_ok(gids: list[str]) -> bool:
            required = 0
            present_ok = 0
            for gid in gids:
                tmpl_gw = template_mw.groups.get(gid)
                if not tmpl_gw:
                    continue
                expected_ratio = float(tmpl_gw.present_ratio)
                if expected_ratio < 0.6:
                    continue
                required += 1
                live_ratio = float((live_mw.groups.get(gid).present_ratio if live_mw.groups.get(gid) else 0.0))
                if live_ratio >= expected_ratio * 0.5:
                    present_ok += 1
            return required == 0 or present_ok >= max(1, required - 1)

        if not (_presence_ok(tracked) or _presence_ok(mirrored_tracked)):
            if self.debug_log_rejections:
                print(f"[REJECT] {gesture_key}: Required groups not sufficiently present")
                print(f"  → Tracked groups: {tracked}")
                for gid in tracked:
                    tmpl_gw = template_mw.groups.get(gid)
                    live_gw = live_mw.groups.get(gid)
                    if tmpl_gw:
                        expected = float(tmpl_gw.present_ratio)
                        actual = float(live_gw.present_ratio) if live_gw else 0.0
                        required_threshold = expected * 0.5
                        status = "✓" if actual >= required_threshold else "✗"
                        print(f"  → {gid}: {status} live={actual:.2f} expected={expected:.2f}")
            return {
                "detected": False,
                "gesture": gesture_key,
                "gesture_name": template.get("name", gesture_key.upper()),
                "score": 0.0,
                "debug": {
                    "method": "v2_metrics",
                    "reason": "required groups not sufficiently present",
                    "tracked_groups": tracked,
                    "eval_frames": len(eval_buffer),
                    "requires_motion": requires_motion,
                    "adaptive_weight": float(adaptive_weight),
                    "adaptive_samples": int(adaptive_samples),
                },
            }

        # Template-driven hand-count gate: prevents 2-hand templates from firing with 1 hand.
        required_hands = int(template.get("required_hands", 0))
        if required_hands > 0:
            hand_groups = [gid for gid in tracked if gid in {"C2", "C4"}]
            if hand_groups:
                min_ratio = float(template.get("hand_presence_min_ratio", 0.45))
                expected_factor = float(template.get("hand_presence_expected_factor", 0.55))
                present_hands = 0
                hand_debug: dict[str, dict[str, float]] = {}

                for gid in hand_groups:
                    live_ratio = float((live_mw.groups.get(gid).present_ratio if live_mw.groups.get(gid) else 0.0))
                    expected_ratio = float((template_mw.groups.get(gid).present_ratio if template_mw.groups.get(gid) else 1.0))
                    required_ratio = max(min_ratio, expected_ratio * expected_factor)
                    hand_debug[gid] = {
                        "live": live_ratio,
                        "required": required_ratio,
                        "expected": expected_ratio,
                    }
                    if live_ratio >= required_ratio:
                        present_hands += 1

                if present_hands < min(required_hands, len(hand_groups)):
                    if self.debug_log_rejections:
                        print(f"[REJECT] {gesture_key}: Insufficient hand presence for template")
                        print(f"  → Required hands: {required_hands}")
                        print(f"  → Present hands: {present_hands}")
                        print(f"  → Hand presence details:")
                        for gid, details in hand_debug.items():
                            status = "✓" if details['live'] >= details['required'] else "✗"
                            print(f"     {gid}: {status} live={details['live']:.2f} required={details['required']:.2f}")
                    return {
                        "detected": False,
                        "gesture": gesture_key,
                        "gesture_name": template.get("name", gesture_key.upper()),
                        "score": 0.0,
                        "debug": {
                            "method": "v2_metrics",
                            "reason": "insufficient hand presence for template",
                            "required_hands": required_hands,
                            "present_hands": present_hands,
                            "hand_presence": hand_debug,
                            "tracked_groups": tracked,
                            "eval_frames": len(eval_buffer),
                            "requires_motion": requires_motion,
                            "adaptive_weight": float(adaptive_weight),
                            "adaptive_samples": int(adaptive_samples),
                        },
                    }

        primary_dbg = {}
        score_primary = motion_window_similarity(live_mw, template_mw, tracked, debug_dict=primary_dbg)
        
        c1_path, c2_path = [], []
        for state in eval_buffer:
            if "C1" in state and state["C1"].anchor_pos:
                c1_path.append(f"{int(state['C1'].anchor_pos[0]*100)}:{int(state['C1'].anchor_pos[1]*100)}")
            if "C2" in state and state["C2"].anchor_pos:
                c2_path.append(f"{int(state['C2'].anchor_pos[0]*100)}:{int(state['C2'].anchor_pos[1]*100)}")
        primary_dbg["path_C1"] = " > ".join(c1_path)
        primary_dbg["path_C2"] = " > ".join(c2_path)

        mirrored_dbg = {}
        mirrored_template = self._build_mirrored_template(template_mw)
        score_mirrored = motion_window_similarity(live_mw, mirrored_template, mirrored_tracked, debug_dict=mirrored_dbg)

        score = max(score_primary, score_mirrored)
        used_mirror = score_mirrored > score_primary

        # JSON-driven motion pattern constraints (no hardcoded gesture name logic).
        wrist_motion = self._compute_bilateral_wrist_motion(eval_buffer)
        pattern = template.get("motion_pattern")
        pattern_score = None
        pattern_blend = 0.0
        if isinstance(pattern, dict) and pattern:
            pattern_score = self._score_motion_pattern(pattern, wrist_motion, live_mw)
            pattern_blend = float(np.clip(pattern.get("blend_weight", 0.35), 0.0, 1.0))
            score = float((1.0 - pattern_blend) * score + pattern_blend * pattern_score)

        if "logic_rules" in template:
            try:
                logic_score = self._score_logic_rules(template["logic_rules"], eval_buffer, primary_dbg)
                score = logic_score
                
                # --- HYBRID EXTENSION ---
                if "spatial_template" in template:
                    spatial_score = self._score_spatial_template(template["spatial_template"], eval_buffer, primary_dbg)
                    spatial_w = float(template["spatial_template"].get("weight", 0.4))
                    score = (score * (1.0 - spatial_w)) + (spatial_score * spatial_w)
                
                score_primary = score
            except Exception as e:
                print(f"EXC LOGIC RULES: {e}")
                import traceback
                traceback.print_exc()
                score = 0.0
            score_mirrored = 0.0

        # Apply an explicit penalty if movement clearly violates the dominant_axis
        dominant_axis = template.get("dominant_axis")
        if not dominant_axis and "motion_type" in template:
            mt = template["motion_type"].lower()
            if "horizontal" in mt: dominant_axis = "horizontal"
            elif "vertical" in mt: dominant_axis = "vertical"
            
        if dominant_axis:
            xs, ys = [], []
            # Extract valid hand coordinates for axis tracking
            for st in eval_buffer:
                for tg in ("C2", "C4"):
                    if tg in st and getattr(st[tg], "anchor_pos", None):
                        xs.append(st[tg].anchor_pos[0])
                        ys.append(st[tg].anchor_pos[1])
            if xs and ys:
                dx = max(xs) - min(xs)
                dy = max(ys) - min(ys)
                # Only apply penalty if the movement span is significant enough
                if max(dx, dy) > 0.03: 
                    if dominant_axis == "horizontal" and dy > dx * 1.3:
                        score *= 0.6  # Penalize severe vertical drift on expected horizontal motion
                        primary_dbg["axis_penalty"] = f"horiz expected, got dy={dy:.2f}/dx={dx:.2f}"
                    elif dominant_axis == "vertical" and dx > dy * 1.3:
                        score *= 0.6  # Penalize severe horizontal drift on expected vertical motion
                        primary_dbg["axis_penalty"] = f"vert expected, got dx={dx:.2f}/dy={dy:.2f}"

        # Use the lower of template threshold and runtime confidence_threshold,
        # so the UI slider can effectively lower detection sensitivity.
        threshold = min(template.get("threshold", self.confidence_threshold), self.confidence_threshold)

        # Determine detection method (peak vs aggregate)
        detected_by_peak = peak_score >= threshold
        detected_by_aggregate = score >= threshold

        if self.debug_log_scoring and score > 0.05:
            print(f"[SCORE] {gesture_key}: agg={score:.3f} peak={peak_score:.3f} (threshold={threshold:.3f})")
            print(f"  → Primary: {score_primary:.3f}, Mirrored: {score_mirrored:.3f}")
            if peak_frame_idx >= 0:
                print(f"  → Peak window centered at frame {peak_frame_idx}/{len(eval_buffer)}")

        return {
            "detected": detected_by_peak or detected_by_aggregate,
            "gesture": gesture_key,
            "gesture_name": template.get("name", gesture_key.upper()),
            "score": float(np.clip(score, 0.0, 1.0)),
            "peak_score": float(np.clip(peak_score, 0.0, 1.0)),
            "peak_frame_idx": peak_frame_idx,
            "detection_method": "peak" if detected_by_peak and not detected_by_aggregate else ("aggregate" if detected_by_aggregate and not detected_by_peak else "both" if detected_by_peak and detected_by_aggregate else "none"),
            "_live_mw": live_mw,
            "debug": {
                "method": "v2_metrics",
                "threshold": threshold,
                "tracked_groups": tracked,
                "score": float(score),
                "score_primary": float(score_primary),
                "score_mirrored": float(score_mirrored),
                "peak_score": float(peak_score),
                "peak_frame_idx": peak_frame_idx,
                "mirrored_match": used_mirror,
                "wrist_motion": wrist_motion,
                "motion_pattern_score": pattern_score,
                "motion_pattern_blend": pattern_blend,
                "eval_frames": len(eval_buffer),
                "requires_motion": requires_motion,
                "adaptive_weight": float(adaptive_weight),
                "adaptive_samples": int(adaptive_samples),
                "score_primary_details": primary_dbg,
            },
        }

    def update(
        self,
        left_hand: Optional[list] = None,
        right_hand: Optional[list] = None,
        pose_landmarks: Optional[list] = None,
        snapshot: Optional[dict] = None,
    ) -> None:
        """
        Update motion tracker with new frame data (dual-hand aware).
        
        Args:
            left_hand: List of 21 (x, y, z) left hand landmarks (pose-verified), or None
            right_hand: List of 21 (x, y, z) right hand landmarks (pose-verified), or None
            pose_landmarks: Optional list of pose landmarks for arm analysis
            snapshot: Optional snapshot dict containing landmark coordinates (used by GroupTracker)
        """
        has_left = bool(left_hand and len(left_hand) > 0)
        has_right = bool(right_hand and len(right_hand) > 0)
        has_any_hand = has_left or has_right

        if has_any_hand:
            self._missing_frame_count = 0
            # Use primary hand for wrist position tracking (prefer right, fallback left)
            primary = right_hand if has_right else left_hand
            wrist_x = primary[0][0]
            wrist_y = primary[0][1]
            self.hand_positions.append((wrist_x, wrist_y))
            self.hand_landmarks_buffer.append(primary)
            self.pose_landmarks_buffer.append(pose_landmarks)

            # Compute group state from snapshot (snapshot carries both hands via extract_snapshot)
            if snapshot:
                self.snapshot_buffer.append(snapshot)
                g_state = self.group_tracker.compute(snapshot)
                self.group_state_buffer.append(g_state)
        else:
            # Tolerate short detector dropouts to keep motion continuity stable.
            self._missing_frame_count += 1
            if self._missing_frame_count > self.max_missing_frames:
                self.hand_positions.clear()
                self.hand_landmarks_buffer.clear()
                self.pose_landmarks_buffer.clear()
                self.group_state_buffer.clear()
                self.snapshot_buffer.clear()
                self.group_tracker.reset()
                self._missing_frame_count = 0

    def set_min_required_frames(self, min_frames: int) -> None:
        """Set minimum frames required before gesture matching"""
        clamped = max(3, min(int(min_frames), self.buffer_size))
        self.min_required_frames = clamped

    def set_buffer_size(self, buffer_size: int) -> None:
        """Set motion buffer size at runtime"""
        new_size = max(5, min(90, int(buffer_size)))
        if new_size == self.buffer_size:
            return

        def _resize(dq: deque) -> deque:
            return deque(list(dq)[-new_size:], maxlen=new_size)

        self.hand_positions = _resize(self.hand_positions)
        self.hand_landmarks_buffer = _resize(self.hand_landmarks_buffer)
        self.pose_landmarks_buffer = _resize(self.pose_landmarks_buffer)
        self.group_state_buffer = _resize(self.group_state_buffer)
        self.snapshot_buffer = _resize(self.snapshot_buffer)

        self.buffer_size = new_size
        self.min_required_frames = max(3, min(self.min_required_frames, self.buffer_size))

    def detect_motion(self, batch_mode: bool = False) -> Dict[str, Any]:
        """
        Detect the best matching motion gesture using V2 metrics.
        
        Returns:
            Detection result dict with:
            - detected: bool
            - gesture: gesture key
            - gesture_name: human-readable name
            - score: 0.0-1.0 confidence
            - debug: diagnostic info
        """
        if not self.action_templates:
            return {
                "detected": False,
                "gesture": "none",
                "gesture_name": "Unknown",
                "score": 0.0,
                "candidates": [],
                "debug": {"reason": "No action templates loaded"}
            }

        if self.debug_logging_enabled:
            print(f"\n[DETECT CYCLE] Templates: {len(self.action_templates)}, Buffer: {len(self.group_state_buffer)} frames")

        best_detected = None
        best_undetected = {
            "detected": False,
            "gesture": "none",
            "gesture_name": "Unknown",
            "score": 0.0,
            "candidates": [],
            "debug": {"reason": "No gesture exceeded threshold"}
        }
        candidates = []

        # Test all templates and find best match
        for gesture_key in sorted(self.action_templates.keys()):
            template = self.action_templates[gesture_key]
            result = self._detect_metric_gesture(gesture_key, template)

            # Extract scoring information for multi-gesture detection
            aggregate_score = result.get("score", 0.0)
            peak_score = result.get("peak_score", 0.0)
            threshold = result.get("debug", {}).get("threshold", self.confidence_threshold)
            detection_method = result.get("detection_method", "none")

            # Determine if this candidate qualifies (peak > 80% OR aggregate > threshold)
            qualifies = peak_score >= 0.80 or aggregate_score >= threshold

            candidates.append({
                "gesture": result.get("gesture"),
                "gesture_name": result.get("gesture_name"),
                "score": aggregate_score,  # Use aggregate for ranking
                "peak_score": peak_score,
                "detected": result.get("detected", False),
                "threshold": threshold,
                "qualifies": qualifies,
                "detection_method": detection_method,
                "peak_frame_idx": result.get("peak_frame_idx", -1),
            })

            if result["detected"]:
                if best_detected is None or result["score"] > best_detected["score"]:
                    best_detected = result
            else:
                if result["score"] > best_undetected["score"]:
                    best_undetected = result

        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        # Prefer detected result; fall back to best undetected for score feedback
        selected = best_detected if best_detected is not None else best_undetected

        # Precision filter 1: require separation from runner-up.
        top1 = candidates[0] if candidates else None
        top2 = candidates[1] if len(candidates) > 1 else None
        top1_score = float(top1.get("score", 0.0)) if top1 else 0.0
        top2_score = float(top2.get("score", 0.0)) if top2 else 0.0
        score_margin = top1_score - top2_score
        margin_ok = (top1 is not None) and (score_margin >= self.decision_margin)

        # Precision filter 2: temporal stability over consecutive cycles.
        # BATCH MODE: bỏ qua stability filter vì buffer đã replay đầy đủ
        
            
        if selected.get("detected") and margin_ok and top1 is not None:
            top1_gesture = top1.get("gesture")
            if top1_gesture == self._stable_gesture:
                self._stable_hits += 1
            else:
                self._stable_gesture = top1_gesture
                self._stable_hits = 1
        else:
            self._stable_gesture = None
            self._stable_hits = 0
            
        if batch_mode:
            stable_ok = True
        else:
            stable_ok = self._stable_hits >= self.stable_required_hits
        
        if selected.get("detected") and (not margin_ok or not stable_ok):
            if self.debug_log_rejections:
                print(f"[REJECT] {selected.get('gesture')}: Precision filter")
                print(f"  → Margin OK: {margin_ok} (margin={score_margin:.3f})")
                print(f"  → Stable OK: {stable_ok} (hits={self._stable_hits})")
            selected = {
                "detected": False,
                "gesture": selected.get("gesture"),
                "gesture_name": selected.get("gesture_name"),
                "score": selected.get("score", 0.0),
                "debug": {
                    **selected.get("debug", {}),
                    "precision_filtered": True,
                    "precision_margin_ok": margin_ok,
                    "precision_margin": float(score_margin),
                    "precision_required_margin": float(self.decision_margin),
                    "precision_stable_hits": int(self._stable_hits),
                    "precision_required_hits": int(self.stable_required_hits),
                },
            }

        # JSON-based detection with no online learning
        if "debug" not in selected:
            selected["debug"] = {}
        selected["debug"]["precision_margin"] = float(score_margin)
        selected["debug"]["precision_margin_ok"] = bool(margin_ok)
        selected["debug"]["precision_stable_hits"] = int(self._stable_hits)
        selected["debug"]["precision_required_hits"] = int(self.stable_required_hits)

        # Remove internal-only fields before returning to caller.
        if "_live_mw" in selected:
            selected.pop("_live_mw", None)

        selected["candidates"] = candidates

        # Update peak records for any candidate scoring ≥ threshold
        frame_count = len(self.group_state_buffer)
        for c in candidates:
            gkey = c.get("gesture")
            sc = float(c.get("score", 0.0) or 0.0)
            if gkey and sc >= self._peak_threshold:
                prev = self._peak_records.get(gkey)
                if prev is None or sc > prev["score"]:
                    self._peak_records[gkey] = {
                        "score": sc,
                        "frame_count": frame_count,
                        "gesture_name": c.get("gesture_name", gkey),
                    }

        if self.debug_logging_enabled:
            print(f"[RESULT] {selected.get('gesture_name')}: detected={selected.get('detected')}, score={selected.get('score', 0.0):.3f}")
            top3_str = ', '.join([f"{c.get('gesture')}:{c.get('score',0):.2f}" for c in candidates[:3]])
            print(f"  → Top 3: {top3_str}")

        return selected

    def set_motion_boost(self, enabled: bool) -> None:
        """Enable/disable motion boost mode for normalized gesture detection."""
        self.motion_boost_enabled = enabled
        if enabled:
            print("[boost] Motion boost enabled: normalizing by body scale and motion cycle")
        else:
            print("[boost] Motion boost disabled")

    def set_gate_fail_factor_override(self, value: Optional[float]) -> None:
        """Override motion-pattern gate penalty factor for all action templates at runtime."""
        if value is None:
            self.gate_fail_factor_override = None
            print("[gate] Gate fail factor override cleared")
            return
        self.gate_fail_factor_override = float(np.clip(value, 0.0, 1.0))
        print(f"[gate] Gate fail factor override: {self.gate_fail_factor_override:.2f}")

    def _estimate_body_scale(self) -> float:
        """
        Estimate body scale from pose landmarks.
        Uses shoulder width as reference (relative to assumed standard width).
        Returns scale factor (1.0 = standard, < 1.0 = smaller person, > 1.0 = larger person)
        """
        if not self.pose_landmarks_buffer or len(self.pose_landmarks_buffer) < 5:
            return 1.0
        
        # Use recent pose landmarks for averaging
        recent_poses = list(self.pose_landmarks_buffer)[-5:]
        widths = []
        
        for pose in recent_poses:
            if pose and len(pose) >= 13:
                # MediaPipe pose: 11=left_shoulder, 12=right_shoulder
                left_shoulder = pose[11]
                right_shoulder = pose[12]
                if len(left_shoulder) >= 2 and len(right_shoulder) >= 2:
                    width = abs(right_shoulder[0] - left_shoulder[0])
                    if width > 20:  # Sanity check
                        widths.append(width)
        
        if not widths:
            return 1.0
        
        avg_width = np.mean(widths)
        # Reference shoulder width in normalized coordinates (~0.4-0.5 of frame width for 640px)
        ref_width = 0.45 * 640  # ~288px
        scale = avg_width / ref_width
        return max(0.6, min(2.0, scale))  # Clamp between 0.6x and 2.0x

    def _estimate_motion_speed(self) -> float:
        """
        Estimate average motion speed from hand position buffer.
        Returns normalized speed (0.0..1.0+ range, 1.0 = reference speed).
        """
        if not self.hand_positions or len(self.hand_positions) < 3:
            return 0.0
        
        positions = list(self.hand_positions)[-10:]  # Recent frames
        distances = []
        
        for i in range(1, len(positions)):
            p1 = positions[i - 1]
            p2 = positions[i]
            dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        avg_dist = np.mean(distances)
        # Reference: ~3 pixels per frame = standard speed (1.0)
        ref_speed = 3.0
        speed = avg_dist / ref_speed
        
        # Use EMA to smooth speed estimate
        alpha = 0.15
        self._motion_speed_avg = alpha * speed + (1 - alpha) * self._motion_speed_avg
        return max(0.0, self._motion_speed_avg)

    def _apply_motion_boost(self, motion_window: MotionWindow) -> MotionWindow:
        """
        Apply boost normalization to motion window.
        Scales angle magnitudes and ratios by body scale and motion speed factors.
        """
        if not self.motion_boost_enabled:
            return motion_window
        
        scale = self._estimate_body_scale()
        speed = self._estimate_motion_speed()
        
        # Boost factors:
        # - Smaller people (scale < 1.0): widen tolerance for angle changes (lower angles expected)
        # - Faster motion (speed > 1.0): tighten tolerance (movements are more pronounced)
        angle_scale = 1.0 / scale  # If person is small (0.7x), angle_scale = 1.42
        speed_scale = 1.0 + 0.3 * (speed - 1.0)  # Speed influence: 0.3 factor
        combined_scale = angle_scale * speed_scale
        
        # Create a normalized copy
        boosted_window = MotionWindow()
        for gid, original_gw in motion_window.groups.items():
            boosted_gw = GroupWindow(
                group_id=gid,
                present_ratio=original_gw.present_ratio,
                angle_mean={k: v * combined_scale for k, v in original_gw.angle_mean.items()},
                angle_std={k: v * combined_scale for k, v in original_gw.angle_std.items()},
                ratio_mean={k: v / scale for k, v in original_gw.ratio_mean.items()},
                ratio_std={k: v / scale for k, v in original_gw.ratio_std.items()},
                dominant_dir=original_gw.dominant_dir,
                dir_consistency=original_gw.dir_consistency,
                spread_mean=(original_gw.spread_mean / scale if original_gw.spread_mean is not None else None),
                spread_std=(original_gw.spread_std / scale if original_gw.spread_std is not None else None),
            )

            boosted_window.groups[gid] = boosted_gw
        
        return boosted_window

    def _get_gesture_display_name(self, gesture_key: str) -> str:
        """Get display name for gesture from manager"""
        gesture = self.gesture_manager.get(gesture_key)
        if gesture:
            return gesture.get("name", gesture_key.upper())
        return gesture_key.upper()
