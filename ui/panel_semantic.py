"""
ui/panel_semantic.py
Standalone semantic accumulation panel (Row 2, right side).
Separated from panel_result so it can occupy its own space.
"""
import cv2
import numpy as np
import time

from ui.theme import (
    FONT, FS_TITLE, FS_BODY, FS_SMALL, FS_TINY,
    FT_BOLD, FT_NORM, LH_BODY, LH_SMALL, LH_TINY,
    BG_PANEL, BG_CARD, COL_BORDER, COL_LABEL, COL_DIM,
    COL_ACTION, COL_LETTER, COL_ACCENT,
    PANEL_HDR_H, draw_panel_header,
)


class PanelSemantic:
    WORDS_MAX = 60

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.words     = []   # (ts, name, score, channel)
        self.threshold = 0.80

    def add(self, name: str, score: float, channel: str):
        """Called from main.py when a gesture is confirmed above threshold."""
        if score < self.threshold:
            return
        if channel not in ("character", "action"):
            return
        if self.words and self.words[-1][1] == name:
            return
        self.words.append((time.time(), name, score, channel))
        if len(self.words) > self.WORDS_MAX:
            self.words.pop(0)

    def set_threshold(self, t: float):
        self.threshold = max(0.0, min(1.0, t))

    def clear(self):
        self.words.clear()

    def render(self, canvas: np.ndarray):
        px, py, w, h = self.x, self.y, self.w, self.h
        cv2.rectangle(canvas, (px, py), (px + w, py + h), BG_PANEL, -1)

        draw_panel_header(canvas, px, py, w,
                          f"SEMANTIC  >{self.threshold*100:.0f}%")

        content_y = py + PANEL_HDR_H + 8
        content_h = h - PANEL_HDR_H - 8

        if not self.words:
            cy = content_y + content_h // 2
            msg = "Waiting for confirmed gestures..."
            tw = cv2.getTextSize(msg, FONT, FS_SMALL, FT_NORM)[0][0]
            cv2.putText(canvas, msg, (px + (w - tw) // 2, cy),
                        FONT, FS_SMALL, COL_DIM, FT_NORM, cv2.LINE_AA)
        else:
            # Render as word chips, newest at bottom
            self._render_chips(canvas, px + 8, content_y, w - 16, content_h)

        # Clear button (bottom right)
        btn_w, btn_h = 52, 22
        btn_x = px + w - btn_w - 6
        btn_y = py + h - btn_h - 6
        cv2.rectangle(canvas, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h),
                      (35, 35, 45), -1)
        cv2.rectangle(canvas, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h),
                      (65, 65, 80), 1)
        cv2.putText(canvas, "CLEAR", (btn_x + 6, btn_y + 15),
                    FONT, FS_TINY, (130, 130, 150), FT_NORM, cv2.LINE_AA)

        cv2.rectangle(canvas, (px, py), (px + w, py + h), (58, 58, 72), 1)

    def _render_chips(self, canvas, x: int, y: int, w: int, h: int):
        """Render word chips in rows, newest at bottom."""
        CHIP_PAD_X = 8
        CHIP_PAD_Y = 4
        CHIP_GAP   = 5
        ROW_H      = LH_BODY + CHIP_PAD_Y * 2 + CHIP_GAP

        # Build chip list newest-last
        chips = []
        for _ts, name, score, channel in self.words:
            col = COL_ACTION if channel == "action" else COL_LETTER
            label = f"[{name}]" if channel == "action" else name
            tw = cv2.getTextSize(label, FONT, FS_BODY, FT_NORM)[0][0]
            chips.append({"label": label, "col": col, "tw": tw})

        # Lay out into rows (left to right)
        rows    = []
        row     = []
        row_w   = 0
        for chip in chips:
            chip_w = chip["tw"] + CHIP_PAD_X * 2
            if row_w + chip_w + CHIP_GAP > w and row:
                rows.append(row)
                row   = [chip]
                row_w = chip_w
            else:
                row.append(chip)
                row_w += chip_w + CHIP_GAP
        if row:
            rows.append(row)

        # Render only as many rows as fit, from bottom
        max_rows = max(1, h // ROW_H)
        visible  = rows[-max_rows:]

        start_y = y + h - len(visible) * ROW_H
        for row in visible:
            cx = x
            for chip in row:
                chip_w = chip["tw"] + CHIP_PAD_X * 2
                chip_h = LH_BODY + CHIP_PAD_Y * 2
                col    = chip["col"]
                bg     = tuple(max(0, c - 160) for c in col)

                cv2.rectangle(canvas, (cx, start_y),
                              (cx + chip_w, start_y + chip_h), bg, -1)
                cv2.rectangle(canvas, (cx, start_y),
                              (cx + chip_w, start_y + chip_h), col, 1)
                cv2.putText(canvas, chip["label"],
                            (cx + CHIP_PAD_X, start_y + chip_h - CHIP_PAD_Y - 2),
                            FONT, FS_BODY, col, FT_NORM, cv2.LINE_AA)
                cx += chip_w + CHIP_GAP
            start_y += ROW_H
