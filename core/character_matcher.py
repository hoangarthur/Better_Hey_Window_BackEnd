"""
Landmark-based static character matcher.
Replaces image silhouette comparison with 3D Finger Joint Kinematics (Angles).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from core.gesture_manager import GestureManager

class CharacterMatcher:
    def __init__(self, gesture_dir: str = "assets/gestures", threshold: float = 0.70):
        self.threshold = threshold
        self.gesture_manager = GestureManager(gesture_dir)
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, dict[str, Any]]:
        templates: dict[str, dict[str, Any]] = {}

        source = {}
        source.update(self.gesture_manager.get_all(category="characters"))
        source.update(self.gesture_manager.get_all(category="numbers"))

        for key, data in source.items():
            points = self._extract_template_points(data)
            if not points:
                continue
            
            angles = self._compute_angles(points)
            if not angles:
                continue

            category = "number" if key.startswith("number_") else "character"
            templates[key] = {
                "key": key,
                "name": data.get("name", key.upper()),
                "category": category,
                "angles": angles,
            }

        return templates

    def _extract_template_points(self, data: dict) -> Optional[list[tuple[float, float, float]]]:
        frames = data.get("frames") or []
        if not frames:
            return None

        fr = frames[0]
        hand_obj = fr.get("left_hand") or fr.get("right_hand")
        if not isinstance(hand_obj, dict):
            return None

        lms = hand_obj.get("landmarks")
        if not isinstance(lms, list) or len(lms) < 21:
            return None

        pts = []
        for item in lms[:21]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return None
            x = float(item[0])
            y = float(item[1])
            z = float(item[2]) if len(item) > 2 else 0.0
            pts.append((x, y, z))
        return pts

    def _compute_angles(self, points: list[tuple[float, float, float]]) -> Optional[list[float]]:
        if len(points) < 21:
            return None

        angles = []

        def vec(p1, p2):
            return (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])

        def angle_between(v1, v2):
            dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            if mag1 < 1e-6 or mag2 < 1e-6:
                return 0.0
            cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            return math.degrees(math.acos(cos_val))

        fingers = [
            [0, 1, 2, 3, 4],     # Thumb
            [0, 5, 6, 7, 8],     # Index
            [0, 9, 10, 11, 12],  # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20]  # Pinky
        ]

        # 1. Bending angles (Flexion) - measure the angle at 3 joints per finger
        for finger in fingers:
            for i in range(len(finger) - 2):
                p1 = points[finger[i]]
                p2 = points[finger[i+1]]
                p3 = points[finger[i+2]]

                v1 = vec(p1, p2)
                v2 = vec(p2, p3)
                ang = angle_between(v1, v2)
                angles.append(ang / 180.0) # Normalize to [0.0, 1.0]

        # 2. Spread angles (Abduction) - measure spread between adjacent fingers
        basal_bones = []
        for finger in fingers:
            # For thumb, using MCP->IP (2->3) reflects pointing direction better
            if finger[1] == 1:
                basal_bones.append(vec(points[2], points[3]))
            else:
                # For others, use MCP->PIP
                basal_bones.append(vec(points[finger[1]], points[finger[2]]))

        for i in range(len(basal_bones) - 1):
            ang = angle_between(basal_bones[i], basal_bones[i+1])
            angles.append(ang / 180.0)

        return angles

    def _angle_distance(self, a: list[float], b: list[float]) -> float:
        # Mean Absolute Error (MAE)
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def _score(self, live_angles: list[float], tmpl: dict[str, Any]) -> float:
        dist = self._angle_distance(live_angles, tmpl["angles"])
        # Chuyển đổi distance lỗi trung bình (0.0 -> 1.0) sang điểm số (score). 
        # Nếu tổng lỗi sai lệnh góc là ~18 độ trung bình mỗi khớp (0.1 normalized), 
        # dist là 0.1, score sẽ là 0.75 cực mịn và hợp lý cho ngưỡng threshold mặc định (0.7).
        return max(0.0, 1.0 - dist * 2.5)

    def match(self, hand_landmarks: Optional[list], threshold: Optional[float] = None) -> dict[str, Any]:
        default_res = {
            "matched": False,
            "gesture": "none",
            "gesture_name": "Unknown",
            "score": 0.0,
            "category": None,
        }

        if not hand_landmarks or len(hand_landmarks) < 21:
            return default_res

        live_points = []
        for item in hand_landmarks[:21]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x = float(item[0])
            y = float(item[1])
            z = float(item[2]) if len(item) > 2 else 0.0
            live_points.append((x, y, z))
            
        if len(live_points) < 21:
            return default_res

        live_angles = self._compute_angles(live_points)
        if not live_angles:
            return default_res

        best_key = None
        best_score = 0.0

        for key, tmpl in self.templates.items():
            score = self._score(live_angles, tmpl)

            if score > best_score:
                best_score = score
                best_key = key

        if not best_key:
            return default_res

        chosen = self.templates[best_key]
        final_threshold = self.threshold if threshold is None else threshold

        return {
            "matched": best_score >= final_threshold,
            "gesture": best_key,
            "gesture_name": chosen["name"],
            "score": best_score,
            "category": chosen["category"],
        }
