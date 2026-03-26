"""
ui/panel_river.py
-----------------
Gesture River — continuous multi-gesture score timeline.

Renders ALL action-gesture scores on a single scrollable river chart,
styled after gesture_debug_overlay.  Each gesture is a coloured line;
labels float at the current-score height on the right edge; any peak
≥ 80 % is pinned with a name chip at that exact position.

Controls (when mouse is in the panel area):
  • Left-drag  → scroll left / right in history
  • Mouse-wheel → scroll (up = back in time)
  • caller may also invoke  scroll_key(+1/-1)
"""

from __future__ import annotations
from collections import deque
import time
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 17-colour palette — one per loaded gesture JSON
_PALETTE: list[tuple[int, int, int]] = [
    (100, 215, 100),   # green
    ( 85, 160, 255),   # blue
    (255, 135,  70),   # orange
    (220,  80, 175),   # pink
    ( 75, 215, 205),   # cyan
    (240, 215,  55),   # yellow
    (170,  95, 248),   # purple
    ( 60, 195, 155),   # teal
    (255,  90,  80),   # red
    (155, 248, 115),   # lime
    (255, 170,  90),   # peach
    (100, 195, 255),   # sky
    (198, 125, 255),   # lavender
    (255, 210, 100),   # gold
    (105, 235, 195),   # mint
    (255, 115, 150),   # salmon
    (150, 235, 250),   # ice
]

PEAK_THRESHOLD = 0.80
HISTORY_LEN    = 2400   # frames per track (~80 s at 30 fps)


class RiverPanel:
    """Scrollable multi-gesture score history chart."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h

        # Per-gesture track  {key: {"name", "color", "scores": deque[float]}}
        self.tracks: dict[str, dict] = {}
        self._color_idx: int = 0

        # Global frame counter (advances on every update() call)
        self.frame_count: int = 0

        # Peaks list  {"key", "name", "score", "frame", "color"}
        self.peaks: list[dict] = []

        # Scroll state (0 = live / right-most view)
        self.scroll_frames: int = 0
        self._drag_x0: int | None = None
        self._drag_scroll0: int = 0
        self.last_interact_t: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, candidates: list[dict]) -> None:
        """
        Feed one time-step.  Call once per render frame.
        Pass the action-candidates list from detect_action(); pass [] when
        no detection is running (time still advances, lines show at 0).
        """
        self.frame_count += 1
        seen: set[str] = set()

        for c in candidates:
            key = c.get("gesture") or ""
            if not key:
                continue
            name  = c.get("gesture_name", key) or key
            score = float(c.get("score", 0.0) or 0.0)
            seen.add(key)

            if key not in self.tracks:
                col = _PALETTE[self._color_idx % len(_PALETTE)]
                self._color_idx += 1
                self.tracks[key] = {
                    "name":   name,
                    "color":  col,
                    "scores": deque(maxlen=HISTORY_LEN),
                }

            self.tracks[key]["scores"].append(score)

            # Peak recording (de-duplicate within a 30-frame window)
            if score >= PEAK_THRESHOLD:
                near = next(
                    (p for p in reversed(self.peaks)
                     if p["key"] == key
                     and abs(self.frame_count - p["frame"]) < 30),
                    None,
                )
                if near is None:
                    self.peaks.append({
                        "key":   key,
                        "name":  name,
                        "score": score,
                        "frame": self.frame_count,
                        "color": self.tracks[key]["color"],
                    })
                elif score > near["score"]:
                    near["score"] = score
                    near["frame"] = self.frame_count

        # Tracks not seen this frame → append 0 (keeps polylines continuous)
        for key, track in self.tracks.items():
            if key not in seen:
                track["scores"].append(0.0)

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        """Route mouse events.  Call from the window's mouse callback."""
        if not (self.y <= y <= self.y + self.h):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_x0        = x
            self._drag_scroll0   = self.scroll_frames

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self._drag_x0 is not None:
                dx = x - self._drag_x0
                view_n = max(1, self.w - 120)          # pixels ≈ frames
                self.scroll_frames = max(
                    0,
                    min(HISTORY_LEN - 50,
                        self._drag_scroll0 + int(-dx * 1.0)),
                )
                self.last_interact_t = time.time()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_x0 = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 60 if flags > 0 else -60
            self.scroll_frames = max(0, min(HISTORY_LEN - 50,
                                            self.scroll_frames + delta))
            self.last_interact_t = time.time()

    def scroll_key(self, direction: int) -> None:
        """
        direction: +1 = look further back in time,
                   -1 = forward (toward live).
        """
        self.scroll_frames = max(0, min(HISTORY_LEN - 50,
                                        self.scroll_frames + direction * 80))
        self.last_interact_t = time.time()

    # ── Render ─────────────────────────────────────────────────────────────

    def render(self, canvas: np.ndarray) -> None:
        rx, ry, rw, rh = self.x, self.y, self.w, self.h

        # ── Panel background ────────────────────────────────────────────
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (10, 10, 10), -1)
        cv2.line(canvas, (rx, ry), (rx + rw, ry), (45, 45, 45), 1)

        # ── Layout ──────────────────────────────────────────────────────
        HDR_H = 22
        MRG_L = 36     # left margin  (y-axis labels)
        MRG_R = 130    # right margin (right-edge labels)
        MRG_T = 8
        MRG_B = 22     # bottom bar

        cx  = rx + MRG_L
        cy  = ry + HDR_H + MRG_T
        cw  = rw - MRG_L - MRG_R
        chh = rh - HDR_H - MRG_T - MRG_B

        is_live = (self.scroll_frames == 0)

        # ── Header ──────────────────────────────────────────────────────
        cv2.putText(canvas, "GESTURE RIVER", (rx + 6, ry + 16),
                    FONT, 0.43, (150, 150, 150), 1, cv2.LINE_AA)

        live_col = (70, 200, 90) if is_live else (210, 130, 60)
        live_txt = ("● LIVE"
                    if is_live
                    else f"◀ -{self.scroll_frames}f   [drag / wheel / [ ] to scroll]")
        cv2.putText(canvas, live_txt, (rx + rw - 290, ry + 16),
                    FONT, 0.38, live_col, 1, cv2.LINE_AA)

        if cw <= 0 or chh <= 0:
            return

        # ── Y-axis grid ──────────────────────────────────────────────────
        for pct, lbl in [(0.25, "25"), (0.5, "50"), (0.75, "75"), (1.0, "100")]:
            gy = int(cy + chh - pct * chh)
            cv2.line(canvas, (cx, gy), (cx + cw, gy), (28, 28, 28), 1)
            cv2.putText(canvas, lbl, (rx + 2, gy + 4),
                        FONT, 0.34, (52, 52, 52), 1, cv2.LINE_AA)

        # 80 % threshold — dashed green
        thr_y = int(cy + chh - 0.80 * chh)
        dash_x = cx
        while dash_x < cx + cw:
            cv2.line(canvas, (dash_x, thr_y),
                     (min(dash_x + 5, cx + cw), thr_y), (40, 130, 40), 1)
            dash_x += 9
        cv2.putText(canvas, "80", (rx + 2, thr_y + 4),
                    FONT, 0.35, (60, 155, 60), 1, cv2.LINE_AA)

        # ── Visible frame window ─────────────────────────────────────────
        view_n  = max(100, cw)            # 1 pixel ≈ 1 frame
        end_f   = self.frame_count - self.scroll_frames
        start_f = end_f - view_n

        def fx(fi: int) -> int:
            return int(cx + (fi - start_f) / max(1, view_n) * cw)

        def fy(sc: float) -> int:
            return int(cy + chh - sc * chh)

        # Prune old peaks
        self.peaks = [p for p in self.peaks
                      if self.frame_count - p["frame"] < HISTORY_LEN]

        # ── Draw gesture lines + collect right-edge labels ────────────────
        right_labels: list[tuple[int, str, tuple]] = []   # (raw_y, text, color)

        for key, track in self.tracks.items():
            scores = track["scores"]
            color  = track["color"]
            name   = track["name"]
            n      = len(scores)
            if n == 0:
                continue

            track_start = self.frame_count - n   # frame index of scores[0]

            # Build polyline for the visible range
            pts: list[tuple[int, int]] = []
            last_score = 0.0

            fi_start = max(start_f, track_start)
            fi_end   = min(end_f + 1, self.frame_count)

            for fi in range(fi_start, fi_end):
                hi = fi - track_start
                if 0 <= hi < n:
                    sc   = scores[hi]
                    px_  = fx(fi)
                    py_  = max(cy, min(cy + chh, fy(sc)))
                    pts.append((px_, py_))
                    last_score = sc

            if len(pts) < 2:
                continue

            # Dim lines that are flat near zero
            brightness = max(0.25, min(1.0, last_score * 4 + 0.15))
            draw_col   = tuple(int(c * brightness) for c in color)
            lw         = 2 if last_score > 0.60 else 1

            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [arr], False, draw_col, lw, cv2.LINE_AA)

            # Right-edge label (only if score is above noise floor and we're live)
            if last_score > 0.10:
                right_labels.append((fy(last_score), f"{name[:12]} {last_score*100:.0f}%", color))

        # ── Stagger right-edge labels (avoid overlap) ────────────────────
        right_labels.sort(key=lambda t: t[0])
        MIN_STEP = 18
        prev_disp_y = -999
        for raw_y, txt, col in right_labels:
            disp_y = max(cy + 8, min(cy + chh - 4, raw_y))
            if disp_y - prev_disp_y < MIN_STEP:
                disp_y = prev_disp_y + MIN_STEP
            disp_y = max(cy + 8, min(cy + chh - 4, disp_y))
            cv2.putText(canvas, txt, (cx + cw + 4, disp_y),
                        FONT, 0.37, col, 1, cv2.LINE_AA)
            prev_disp_y = disp_y

        # ── Peak chips ───────────────────────────────────────────────────
        placed: dict[tuple[int, int], bool] = {}   # (x_bucket, y_bucket)

        for peak in self.peaks:
            pf = peak["frame"]
            if not (start_f <= pf <= end_f):
                continue

            px_  = fx(pf)
            py_  = fy(peak["score"])

            if not (cx <= px_ <= cx + cw):
                continue

            pcol   = peak["color"]
            plabel = f"{peak['name'][:10]} {peak['score']*100:.0f}%"
            (tw_, th_), _ = cv2.getTextSize(plabel, FONT, 0.38, 1)

            # Position chip above dot
            chip_x = max(cx, min(cx + cw - tw_ - 6, px_ - tw_ // 2))
            chip_y = py_ - th_ - 6
            chip_y = max(cy + 2, chip_y)

            # Avoid chip-on-chip overlap (shift down by 16 px slots)
            xb = chip_x // 50
            yb = chip_y // 16
            offset = 0
            while (xb, (chip_y + offset) // 16) in placed:
                offset += 16
            chip_y = min(cy + chh - th_ - 2, chip_y + offset)
            placed[(xb, chip_y // 16)] = True

            # Dot at peak
            cv2.circle(canvas, (px_, py_), 3, pcol, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px_, py_), 3, (255, 255, 255), 1, cv2.LINE_AA)

            # Dotted connector from chip bottom to dot
            dot_y = chip_y + th_ + 4
            while dot_y < py_ - 2:
                cv2.circle(canvas, (px_, dot_y), 1, (65, 65, 65), -1)
                dot_y += 4

            # Chip background + border + text
            cv2.rectangle(canvas,
                          (chip_x - 2, chip_y - 1),
                          (chip_x + tw_ + 4, chip_y + th_ + 2),
                          (20, 20, 20), -1)
            cv2.rectangle(canvas,
                          (chip_x - 2, chip_y - 1),
                          (chip_x + tw_ + 4, chip_y + th_ + 2),
                          pcol, 1)
            cv2.putText(canvas, plabel, (chip_x, chip_y + th_),
                        FONT, 0.38, pcol, 1, cv2.LINE_AA)

        # ── Chart border ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + chh), (38, 38, 38), 1)

        # ── Bottom info bar ──────────────────────────────────────────────
        n_vis_peaks = sum(1 for p in self.peaks if start_f <= p["frame"] <= end_f)
        bar_y = cy + chh + 12
        info  = (f"frames {max(0, start_f)}–{end_f}  "
                 f"total {self.frame_count}  "
                 f"peaks in view: {n_vis_peaks}  "
                 f"tracks: {len(self.tracks)}")
        cv2.putText(canvas, info, (cx, bar_y),
                    FONT, 0.34, (55, 55, 55), 1, cv2.LINE_AA)