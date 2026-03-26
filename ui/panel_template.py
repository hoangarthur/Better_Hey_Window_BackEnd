"""
ui/panel_template.py  —  v2
Panel 2: Template keypoint visualiser + match result.
Larger fonts, cleaner candidate list, better score bar.
"""
import cv2
import numpy as np
import time
from core.renderer import render_template_frame, render_metric_pose
from ui.theme import (
    FONT, FS_HERO, FS_TITLE, FS_BODY, FS_SMALL, FS_TINY,
    FT_BOLD, FT_NORM, LH_BODY, LH_SMALL,
    BG_PANEL, BG_CARD, COL_BORDER, COL_LABEL, COL_TEXT, COL_DIM,
    COL_ACTION, COL_LETTER, COL_ACCENT, COL_WARN, COL_FAIL,
    PANEL_HDR_H, draw_panel_header, score_color, draw_score_bar,
)

ANIM_FPS   = 8
COL_MATCH  = COL_ACTION
COL_ORANGE = (255, 105, 50)


class PanelTemplate:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.template          = None
        self.frame_idx         = 0
        self.last_tick         = time.time()
        self.match_flash       = 0.0
        self.match_score       = 0.0
        self.action_name       = "—"
        self.action_score      = 0.0
        self.character_name    = "—"
        self.character_score   = 0.0
        self.metric_anim_phase = 0.0
        self.metric_last_tick  = time.time()
        self.debug_lines       = []
        self.candidates        = []

    # ── Public API ─────────────────────────────────────────────────────────

    def set_template(self, template):
        self.template   = template
        self.frame_idx  = 0
        self.match_score = 0.0
        self.candidates = []

    def set_match_score(self, score: float):
        self.match_score = max(0.0, min(1.0, score))
        if score >= 0.70:
            self.match_flash = time.time()

    def notify_match(self):
        self.match_flash = time.time()

    def update_match_result(self, action_name, action_score,
                            character_name, character_score):
        self.action_name    = action_name or "—"
        self.action_score   = max(0.0, min(1.0, action_score))
        self.character_name = character_name or "—"
        self.character_score = max(0.0, min(1.0, character_score))

    def set_debug_lines(self, lines):
        self.debug_lines = list(lines or [])[:4]

    def update_candidates(self, candidates):
        filtered = [c for c in candidates
                    if c.get("score", 0.0) > 0.60]
        self.candidates = sorted(
            filtered, key=lambda x: x.get("score", 0.0), reverse=True)[:6]

    def get_preview_rect(self):
        content_y = self.y + PANEL_HDR_H
        top_h = int((self.h - PANEL_HDR_H) * 0.50)
        return self.x, content_y, self.w, top_h

    # ── Render ─────────────────────────────────────────────────────────────

    def render(self, canvas: np.ndarray):
        px, py = self.x, self.y
        content_y = py + PANEL_HDR_H
        top_h     = int((self.h - PANEL_HDR_H) * 0.50)
        bottom_y  = content_y + top_h
        bottom_h  = self.h - PANEL_HDR_H - top_h

        cv2.rectangle(canvas, (px, content_y), (px + self.w, py + self.h),
                      BG_PANEL, -1)

        # ── Template preview (top half) ───────────────────────────────────
        if self.template and self.template.get("frames"):
            self._advance_frame()
            snap    = self.template["frames"][self.frame_idx]
            dot_img = render_template_frame(snap, self.w, top_h)
            self._apply_flash(dot_img)
            canvas[content_y:content_y + top_h, px:px + self.w] = dot_img
            name = self.template.get("name", "")
            cv2.putText(canvas, name, (px + 10, py + PANEL_HDR_H - 4),
                        FONT, FS_BODY, COL_ORANGE, FT_NORM, cv2.LINE_AA)
            total = len(self.template.get("frames", []))
            cv2.putText(canvas,
                        f"{self.frame_idx + 1}/{total}",
                        (px + self.w - 52, py + PANEL_HDR_H - 4),
                        FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)

        elif self.template:
            self._advance_metric_phase()
            pose_img = render_metric_pose(
                self.template, self.metric_anim_phase, self.w, top_h)
            self._apply_flash(pose_img)
            canvas[content_y:content_y + top_h, px:px + self.w] = pose_img
            name = self.template.get("name", "Unknown Action")
            cv2.putText(canvas, name, (px + 10, py + PANEL_HDR_H - 4),
                        FONT, FS_BODY, COL_ORANGE, FT_NORM, cv2.LINE_AA)
        else:
            _draw_empty(canvas, px, content_y, self.w, top_h)

        # ── Divider ───────────────────────────────────────────────────────
        cv2.line(canvas, (px + 8, bottom_y), (px + self.w - 8, bottom_y),
                 (38, 38, 50), 1)

        # ── Bottom half: scores + candidates ─────────────────────────────
        y = bottom_y + 14

        # Score bar + hero number
        BAR_H = 12
        draw_score_bar(canvas, px + 10, y, self.w - 20, BAR_H,
                       self.match_score, score_color(self.match_score))
        y += BAR_H + 6

        pct_str = f"{self.match_score * 100:.1f}%"
        hero_scale = FS_HERO
        tw = cv2.getTextSize(pct_str, FONT, hero_scale, FT_BOLD)[0][0]
        col = score_color(self.match_score)
        cv2.putText(canvas, pct_str,
                    (px + (self.w - tw) // 2, y + 32),
                    FONT, hero_scale, col, FT_BOLD, cv2.LINE_AA)
        y += 40

        # ACTION / LETTER row
        col_w = (self.w - 28) // 2
        col_x1 = px + 10
        col_x2 = col_x1 + col_w + 8
        cv2.line(canvas, (col_x2 - 4, y - 4), (col_x2 - 4, y + LH_BODY + LH_SMALL + 4),
                 (40, 40, 52), 1)

        cv2.putText(canvas, "ACTION", (col_x1, y),
                    FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)
        cv2.putText(canvas, "LETTER", (col_x2, y),
                    FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)
        y += LH_SMALL

        col_a = COL_ACTION if self.action_score > 0 else COL_DIM
        col_l = COL_LETTER if self.character_score > 0 else COL_DIM
        cv2.putText(canvas, self.action_name[:14], (col_x1, y),
                    FONT, FS_BODY, col_a,
                    FT_BOLD if self.action_score > 0 else FT_NORM, cv2.LINE_AA)
        cv2.putText(canvas, self.character_name[:14], (col_x2, y),
                    FONT, FS_BODY, col_l,
                    FT_BOLD if self.character_score > 0 else FT_NORM, cv2.LINE_AA)
        y += LH_SMALL

        if self.action_score > 0:
            cv2.putText(canvas, f"{self.action_score*100:.0f}%", (col_x1, y),
                        FONT, FS_TINY, col_a, FT_NORM, cv2.LINE_AA)
        if self.character_score > 0:
            cv2.putText(canvas, f"{self.character_score*100:.0f}%", (col_x2, y),
                        FONT, FS_TINY, col_l, FT_NORM, cv2.LINE_AA)
        y += LH_BODY

        # Candidates list
        if self.candidates:
            cv2.putText(canvas, "CANDIDATES", (px + 10, y),
                        FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)
            y += LH_SMALL

            for cand in self.candidates:
                if y + LH_SMALL > py + self.h - 6:
                    break
                name  = cand.get("gesture_name", "?")[:18]
                score = cand.get("score", 0.0)
                col   = score_color(score)
                # Bar behind name
                bar_w = max(4, int((self.w - 24) * score))
                cv2.rectangle(canvas,
                               (px + 10, y - LH_SMALL + 4),
                               (px + 10 + bar_w, y + 2),
                               (28, 28, 36), -1)
                fill = max(2, int(bar_w * score))
                cv2.rectangle(canvas,
                               (px + 10, y - LH_SMALL + 4),
                               (px + 10 + fill, y + 2),
                               (*col[:3], 80) if len(col) > 3 else col, -1)
                # Text
                cv2.putText(canvas, name, (px + 14, y),
                            FONT, FS_SMALL, col, FT_NORM, cv2.LINE_AA)
                pct_x = px + self.w - 42
                cv2.putText(canvas, f"{score*100:.0f}%", (pct_x, y),
                            FONT, FS_SMALL, col, FT_NORM, cv2.LINE_AA)
                y += LH_SMALL

        elif self.debug_lines:
            for line in self.debug_lines:
                if y + LH_SMALL > py + self.h - 6:
                    break
                cv2.putText(canvas, line[:36], (px + 10, y),
                            FONT, FS_TINY, (115, 115, 130), FT_NORM, cv2.LINE_AA)
                y += LH_SMALL

        # ── Border + header ───────────────────────────────────────────────
        cv2.rectangle(canvas, (px, py), (px + self.w, py + self.h),
                      (58, 58, 72), 1)
        draw_panel_header(canvas, px, py, self.w, "02  TEMPLATE  REFERENCE")

    # ── Internal ───────────────────────────────────────────────────────────

    def _apply_flash(self, img):
        age = time.time() - self.match_flash
        if age < 0.55:
            alpha = 1.0 - age / 0.55
            overlay = np.zeros_like(img)
            overlay[:] = (int(50 * alpha), int(245 * alpha), int(80 * alpha))
            cv2.addWeighted(img, 1.0, overlay, 0.22, 0, img)

    def _advance_frame(self):
        if not self.template or not self.template.get("frames"):
            return
        now = time.time()
        if now - self.last_tick >= 1.0 / ANIM_FPS:
            self.frame_idx = (self.frame_idx + 1) % len(self.template["frames"])
            self.last_tick = now

    def _advance_metric_phase(self):
        now = time.time()
        dt  = now - self.metric_last_tick
        self.metric_last_tick = now
        self.metric_anim_phase = (self.metric_anim_phase + dt * 0.5) % 1.0


def _draw_empty(canvas, x, y, w, h):
    msg = "No template"
    tw  = cv2.getTextSize(msg, FONT, FS_SMALL, FT_NORM)[0][0]
    cv2.putText(canvas, msg,
                (x + (w - tw) // 2, y + h // 2 + 6),
                FONT, FS_SMALL, (50, 50, 60), FT_NORM, cv2.LINE_AA)
