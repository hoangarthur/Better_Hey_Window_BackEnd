"""
ui/panel_live.py  —  v2
Panel 1: Live camera feed split into Face / Left Hand / Right Hand / Pose zones.
Larger fonts, cleaner zone borders, landmark count indicators.
"""
import cv2
import numpy as np
from core.detector import draw_face_zone, draw_left_hand, draw_right_hand, draw_pose_zone
from ui.theme import (
    FONT, FS_TITLE, FS_BODY, FS_SMALL, FS_TINY,
    FT_BOLD, FT_NORM, BG_PANEL, COL_BORDER, COL_LABEL,
    PANEL_HDR_H, draw_panel_header,
)

COL_FACE = ( 0, 210, 245)
COL_HAND = ( 0, 245, 135)
COL_POSE = (255, 175,  30)


class PanelLive:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self._calc_zones()

    def _calc_zones(self):
        content_h = self.h - PANEL_HDR_H
        face_h    = int(content_h * 0.18)
        hand_h    = int(content_h * 0.36)
        pose_h    = content_h - face_h - hand_h
        half_w    = self.w // 2

        self.zones = {
            "face":        (0,                 0,        self.w, face_h),
            "left_hand":   (0,                 face_h,   half_w, hand_h),
            "right_hand":  (half_w,            face_h,   half_w, hand_h),
            "pose":        (0,                 face_h + hand_h, self.w, pose_h),
        }

    def render(self, canvas: np.ndarray, mp_results, cam_frame=None):
        px, py = self.x, self.y
        content_y = py + PANEL_HDR_H

        for zone_key, (zx, zy, zw, zh) in self.zones.items():
            ax, ay = px + zx, content_y + zy
            cv2.rectangle(canvas, (ax, ay), (ax + zw, ay + zh), BG_PANEL, -1)

            inner = (ax, ay, zw, zh)
            if mp_results:
                if zone_key == "face":
                    draw_face_zone(canvas, mp_results, inner)
                elif zone_key == "left_hand":
                    draw_left_hand(canvas, mp_results, inner)
                elif zone_key == "right_hand":
                    draw_right_hand(canvas, mp_results, inner)
                elif zone_key == "pose":
                    draw_pose_zone(canvas, mp_results, inner)

            # Zone label + border
            accent = {"face": COL_FACE, "left_hand": COL_HAND,
                      "right_hand": COL_HAND, "pose": COL_POSE}.get(zone_key, COL_HAND)
            label  = {"face": "FACE", "left_hand": "LEFT",
                      "right_hand": "RIGHT", "pose": "BODY"}.get(zone_key, "")

            cv2.rectangle(canvas, (ax, ay), (ax + zw, ay + zh),
                          (38, 38, 48), 1)
            cv2.putText(canvas, label, (ax + 8, ay + 18),
                        FONT, FS_SMALL, accent, FT_NORM, cv2.LINE_AA)

            # Landmark present indicator dot (top-right of zone)
            present = _zone_has_landmarks(mp_results, zone_key)
            dot_col = (55, 200, 80) if present else (80, 40, 40)
            cv2.circle(canvas, (ax + zw - 10, ay + 10), 4, dot_col, -1)

        cv2.rectangle(canvas, (px, py), (px + self.w, py + self.h), (58, 58, 72), 1)
        draw_panel_header(canvas, px, py, self.w, "01  LIVE DETECTION")


def _zone_has_landmarks(mp_results, zone_key: str) -> bool:
    if not mp_results:
        return False
    if zone_key == "face":
        return bool(mp_results.get("face_landmarks"))
    elif zone_key == "left_hand":
        return bool(mp_results.get("left_hand_landmarks"))
    elif zone_key == "right_hand":
        return bool(mp_results.get("right_hand_landmarks"))
    elif zone_key == "pose":
        return bool(mp_results.get("pose_landmarks"))
    return False
