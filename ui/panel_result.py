"""
ui/panel_result.py  —  v3
Panel 3: Scan Log (card-based, scrollable) + Status bar

Each scan session renders as a card with:
  - Left accent border: green=accepted / red=failed / amber=partial
  - SCAN block:   candidate listing with mini score bars
  - VERIFY block: rescan pass/fail results
  - RESULT block: what was accepted (yellow) or nothing (dim red)

Scroll: mouse wheel anywhere over the panel, or call scroll(delta).
"""
import cv2
import numpy as np
import time
import math

from ui.theme import (
    FONT, FS_TITLE, FS_BODY, FS_SMALL, FS_TINY,
    FT_BOLD, FT_NORM, LH_BODY, LH_SMALL, LH_TINY,
    BG_PANEL, BG_CARD, BG_HDR, COL_BORDER, COL_BORDER_HI,
    COL_LABEL, COL_TEXT, COL_DIM, COL_BRIGHT,
    COL_ACTION, COL_LETTER, COL_ACCENT, COL_PASS, COL_FAIL,
    COL_WARN, COL_STATUS, COL_CANDIDATE,
    COL_CARD_OK, COL_CARD_FAIL, COL_CARD_WARN,
    PANEL_HDR_H, CARD_PAD, CARD_GAP, SCROLLBAR_W,
    draw_panel_header, score_color, draw_score_bar,
)


class PanelResult:
    MAX_STREAM_LOG     = 20
    SEMANTIC_WORDS_MAX = 40

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h

        # Detection state
        self.score           = 0.0
        self.matched         = False
        self.action_name     = "—"
        self.action_score    = 0.0
        self.character_name  = "—"
        self.character_score = 0.0
        self.decision        = "Ready"
        self.progress        = 0.0

        # Semantic
        self.semantic_threshold = 0.80
        self.semantic_words  = []    # (ts, name, score, channel)

        # Scan log
        self.stream_log      = []
        self.cache_history   = []
        self.history         = []

        # Scroll state (number of cards to skip from newest)
        self.scroll_offset   = 0
        self._pulse          = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, score: float, matched: bool, template: dict | None):
        self.score   = score
        self.matched = matched
        if matched and template:
            name    = template.get("name", "Unknown")
            channel = template.get("channel", "character")
            if channel == "action":
                self.action_name  = name
                self.action_score = score
            else:
                self.character_name  = name
                self.character_score = score
            if not self.history or self.history[-1][1] != name:
                self.history.append((time.strftime("%H:%M:%S"), name, score, channel))
                if len(self.history) > 20:
                    self.history.pop(0)

    def add_to_semantic(self, name: str, score: float, channel: str):
        if score >= self.semantic_threshold and channel in ("character", "action"):
            if not self.semantic_words or self.semantic_words[-1][1] != name:
                self.semantic_words.append((time.time(), name, score, channel))
                if len(self.semantic_words) > self.SEMANTIC_WORDS_MAX:
                    self.semantic_words.pop(0)

    def get_semantic_items(self):
        items = []
        for _ts, name, _sc, channel in self.semantic_words:
            items.append(f"[{name}]" if channel == "action" else name)
        return items

    def set_semantic_threshold(self, t: float):
        self.semantic_threshold = max(0.0, min(1.0, t))

    def set_decision(self, text: str):
        self.decision = text

    def add_to_cache_history(self, name, score, channel="action"):
        if self.cache_history and self.cache_history[-1][0] == name:
            return
        self.cache_history.append((name, score))
        if len(self.cache_history) > 20:
            self.cache_history.pop(0)

    def scroll(self, delta: int):
        """Scroll log by delta cards (positive = back in time)."""
        max_off = max(0, len(self.stream_log) - 1)
        self.scroll_offset = max(0, min(max_off, self.scroll_offset + delta))

    # ── Scan log session API ───────────────────────────────────────────────

    def begin_scan_session(self, peaks: dict) -> None:
        sorted_peaks = sorted(
            peaks.items(), key=lambda kv: kv[1]["score"], reverse=True
        )
        session = {
            "ts":       time.strftime("%H:%M:%S"),
            "peaks":    [{"name": k, "score": v["score"]} for k, v in sorted_peaks],
            "rescans":  [],
            "accepted": [],
        }
        self.stream_log.append(session)
        if len(self.stream_log) > self.MAX_STREAM_LOG:
            self.stream_log.pop(0)
        self.scroll_offset = 0   # jump to newest

    def log_rescan_result(self, name: str | None, pass_num: int,
                          confirmed: bool, score: float) -> None:
        if not name or not self.stream_log:
            return
        session = self.stream_log[-1]
        session["rescans"].append({
            "name": name, "pass": pass_num,
            "ok": confirmed, "score": score,
        })
        if confirmed:
            session["accepted"].append({"name": name, "score": score})

    # ── Render ─────────────────────────────────────────────────────────────

    def render(self, canvas: np.ndarray):
        px, py, w, h = self.x, self.y, self.w, self.h

        # Panel background
        cv2.rectangle(canvas, (px, py), (px + w, py + h), BG_PANEL, -1)

        self._pulse = (self._pulse + 0.04) % (2 * math.pi)
        pulse = 0.5 + 0.5 * math.sin(self._pulse)

        # ── Fixed header ──────────────────────────────────────────────────
        draw_panel_header(canvas, px, py, w, "03  SCAN LOG  +  SEMANTIC")

        content_y = py + PANEL_HDR_H + 4

        # ── Status bar (fixed, ~40px) ─────────────────────────────────────
        STATUS_H = 42
        self._render_status(canvas, px, content_y, w, STATUS_H)
        content_y += STATUS_H

        # ── Progress bar (only during ANALYZING) ─────────────────────────
        if self.progress > 0:
            PROG_H = 5
            cv2.rectangle(canvas, (px, content_y),
                          (px + w, content_y + PROG_H), (30, 30, 38), -1)
            fill = max(4, int(w * self.progress))
            prog_col = COL_ACTION if self.progress >= 1.0 else COL_WARN
            cv2.rectangle(canvas, (px, content_y),
                          (px + fill, content_y + PROG_H), prog_col, -1)
            content_y += PROG_H + 2

        # ── Semantic strip (fixed bottom, ~52px) ──────────────────────────
        SEMANTIC_H = 52
        semantic_y = py + h - SEMANTIC_H
        cv2.line(canvas, (px + 8, semantic_y), (px + w - 8, semantic_y),
                 (35, 35, 45), 1)
        self._render_semantic(canvas, px, semantic_y + 6, w, SEMANTIC_H - 6)

        # ── Scrollable scan log ───────────────────────────────────────────
        log_bottom = semantic_y - 4
        self._render_scan_log(canvas, px, content_y, w, log_bottom)

        # ── Pulse dot + outer border ──────────────────────────────────────
        dot_r = int(3 + 2 * pulse)
        dot_col = (45, int(165 + 70 * pulse), 45)
        cv2.circle(canvas, (px + w - 12, py + PANEL_HDR_H - 10),
                   dot_r, dot_col, -1)
        cv2.rectangle(canvas, (px, py), (px + w, py + h), (58, 58, 72), 1)

    # ── Status bar ─────────────────────────────────────────────────────────

    def _render_status(self, canvas, px: int, py: int, w: int, h: int):
        # Thin card background
        cv2.rectangle(canvas, (px + 4, py), (px + w - 4, py + h - 4),
                      (17, 18, 24), -1)

        mid = py + h // 2

        # ACTION left half
        col_a = COL_ACTION if self.action_score > 0 else COL_DIM
        cv2.putText(canvas, "ACTION", (px + 12, mid - 8),
                    FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)
        a_name = (self.action_name[:18] if self.action_name != "—"
                  else "—")
        cv2.putText(canvas, a_name, (px + 12, mid + 10),
                    FONT, FS_BODY, col_a, FT_BOLD if self.action_score > 0 else FT_NORM,
                    cv2.LINE_AA)
        if self.action_score > 0:
            cv2.putText(canvas, f"{self.action_score*100:.0f}%",
                        (px + 12, mid + 24),
                        FONT, FS_TINY, col_a, FT_NORM, cv2.LINE_AA)

        # Divider
        mid_x = px + w // 2
        cv2.line(canvas, (mid_x, py + 6), (mid_x, py + h - 8),
                 (40, 40, 52), 1)

        # LETTER right half
        col_l = COL_LETTER if self.character_score > 0 else COL_DIM
        cv2.putText(canvas, "LETTER", (mid_x + 10, mid - 8),
                    FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)
        l_name = (self.character_name[:18] if self.character_name != "—"
                  else "—")
        cv2.putText(canvas, l_name, (mid_x + 10, mid + 10),
                    FONT, FS_BODY, col_l,
                    FT_BOLD if self.character_score > 0 else FT_NORM,
                    cv2.LINE_AA)
        if self.character_score > 0:
            cv2.putText(canvas, f"{self.character_score*100:.0f}%",
                        (mid_x + 10, mid + 24),
                        FONT, FS_TINY, col_l, FT_NORM, cv2.LINE_AA)

        # Decision line bottom-right
        if self.decision:
            tw = cv2.getTextSize(self.decision, FONT, FS_TINY, FT_NORM)[0][0]
            cv2.putText(canvas, self.decision,
                        (px + w - tw - 10, py + h - 6),
                        FONT, FS_TINY, COL_STATUS, FT_NORM, cv2.LINE_AA)

    # ── Scan log ───────────────────────────────────────────────────────────

    def _render_scan_log(self, canvas, px: int, py: int,
                         w: int, y_bottom: int):
        INDENT  = px + 18
        BAR_W   = max(60, w // 6)
        CARD_W  = w - 12

        if not self.stream_log:
            cv2.putText(canvas, "No sessions yet — perform a gesture",
                        (INDENT, py + LH_SMALL + 4),
                        FONT, FS_SMALL, COL_DIM, FT_NORM, cv2.LINE_AA)
            return

        # Build render list (newest first, respecting scroll_offset)
        ordered = list(reversed(self.stream_log))
        start   = self.scroll_offset
        visible = ordered[start:]

        y     = py + 4
        drawn = 0

        for session in visible:
            card_h = self._session_card_height(session)
            if y + card_h > y_bottom:
                break
            self._draw_session_card(canvas, px + 6, y, CARD_W, card_h,
                                    session, INDENT - px - 6, BAR_W)
            y += card_h + CARD_GAP
            drawn += 1

        # Scrollbar
        total = len(self.stream_log)
        if total > 1 and drawn < total:
            sb_h = y_bottom - py
            thumb_h = max(20, int(sb_h * drawn / total))
            thumb_y = py + int((sb_h - thumb_h) * self.scroll_offset / max(1, total - 1))
            sb_x = px + w - SCROLLBAR_W - 2
            cv2.rectangle(canvas, (sb_x, py), (sb_x + SCROLLBAR_W, py + sb_h),
                          (28, 28, 36), -1)
            cv2.rectangle(canvas, (sb_x, thumb_y),
                          (sb_x + SCROLLBAR_W, thumb_y + thumb_h),
                          (72, 72, 92), -1)

        # Scroll hint
        if self.scroll_offset > 0:
            cv2.putText(canvas, f"^ {self.scroll_offset} older",
                        (px + 10, py + 14),
                        FONT, FS_TINY, (72, 72, 90), FT_NORM, cv2.LINE_AA)

    def _session_card_height(self, session: dict) -> int:
        """Estimate card height before drawing."""
        lines = 1                                      # header line
        lines += min(len(session["peaks"]), 5)         # scan candidates
        lines += len(session["rescans"])               # verify passes
        lines += 1 if session["accepted"] or session["rescans"] else 0
        return CARD_PAD * 2 + lines * LH_SMALL + 4

    def _draw_session_card(self, canvas, cx: int, cy: int,
                           cw: int, ch: int, session: dict,
                           indent_off: int, bar_w: int):
        """Draw one session card at (cx, cy) with size (cw, ch)."""
        accepted  = bool(session.get("accepted"))
        any_fail  = any(not r["ok"] for r in session.get("rescans", []))

        # Background + border
        bg  = (19, 22, 28) if accepted else (22, 16, 16) if (any_fail and not accepted) else (20, 20, 14)
        bdr = (42, 44, 56)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), bg, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), bdr, 1)

        # Left accent stripe
        if accepted:
            stripe = COL_CARD_OK
        elif any_fail and not accepted:
            stripe = COL_CARD_FAIL
        else:
            stripe = COL_CARD_WARN
        cv2.rectangle(canvas, (cx, cy), (cx + 3, cy + ch), stripe, -1)

        ix = cx + indent_off + 8    # text indent
        y  = cy + CARD_PAD

        # ── Header: timestamp + candidate count ──────────────────────────
        n   = len(session["peaks"])
        hdr = f"{session['ts']}   {n} candidate{'s' if n != 1 else ''}"
        cv2.putText(canvas, hdr, (ix, y + LH_SMALL - 4),
                    FONT, FS_SMALL, (155, 155, 172), FT_NORM, cv2.LINE_AA)
        y += LH_SMALL + 2

        # ── SCAN block ────────────────────────────────────────────────────
        cv2.putText(canvas, "SCAN", (ix, y + LH_TINY - 2),
                    FONT, FS_TINY, (80, 80, 95), FT_NORM, cv2.LINE_AA)
        y += LH_TINY

        for i, peak in enumerate(session["peaks"][:5]):
            pname  = peak["name"][:20]
            psc    = peak["score"]
            is_top = (i == 0)

            # Mini bar
            draw_score_bar(canvas, ix, y - LH_SMALL + 4,
                           bar_w, LH_SMALL - 6, psc)

            # Name + score
            col = (125, 220, 135) if is_top else (95, 95, 112)
            sym = "●" if is_top else "○"
            cv2.putText(canvas, f"{sym} {pname}",
                        (ix + bar_w + 6, y),
                        FONT, FS_SMALL, col, FT_NORM, cv2.LINE_AA)
            pct_x = cx + cw - 38
            cv2.putText(canvas, f"{psc*100:.0f}%", (pct_x, y),
                        FONT, FS_SMALL, col, FT_NORM, cv2.LINE_AA)
            y += LH_SMALL

        # ── VERIFY block ──────────────────────────────────────────────────
        if session["rescans"]:
            y += 2
            cv2.putText(canvas, "VERIFY", (ix, y + LH_TINY - 2),
                        FONT, FS_TINY, (80, 80, 95), FT_NORM, cv2.LINE_AA)
            y += LH_TINY

            for r in session["rescans"]:
                ok_col = COL_PASS if r["ok"] else COL_FAIL
                sym    = "+" if r["ok"] else "-"
                sc_str = f"{r['score']*100:.0f}%" if r["ok"] else "—"
                line   = f"P{r['pass']}  {r['name'][:18]}  {sc_str}"
                # Colored marker box
                cv2.rectangle(canvas,
                               (ix, y - LH_SMALL + 4),
                               (ix + 10, y + 2), ok_col, -1)
                cv2.putText(canvas, line, (ix + 14, y),
                            FONT, FS_SMALL, (165, 165, 180), FT_NORM, cv2.LINE_AA)
                y += LH_SMALL

        # ── RESULT block ──────────────────────────────────────────────────
        if session["accepted"]:
            y += 2
            items = "  +  ".join(
                f"{a['name'][:16]}  {a['score']*100:.0f}%"
                for a in session["accepted"]
            )
            cv2.putText(canvas, f"TOOK  {items}", (ix, y + LH_SMALL - 4),
                        FONT, FS_SMALL, (185, 165, 60), FT_NORM, cv2.LINE_AA)
        elif session["rescans"]:
            y += 2
            cv2.putText(canvas, "TOOK  —  no match", (ix, y + LH_SMALL - 4),
                        FONT, FS_SMALL, (155, 65, 65), FT_NORM, cv2.LINE_AA)

    # ── Semantic ────────────────────────────────────────────────────────────

    def _render_semantic(self, canvas, px: int, py: int, w: int, h: int):
        items = self.get_semantic_items()
        cv2.putText(canvas, f"SEMANTIC  >{self.semantic_threshold*100:.0f}%",
                    (px + 10, py + LH_TINY),
                    FONT, FS_TINY, COL_LABEL, FT_NORM, cv2.LINE_AA)

        if not items:
            cv2.putText(canvas, "—", (px + 10, py + LH_TINY + LH_SMALL),
                        FONT, FS_SMALL, COL_DIM, FT_NORM, cv2.LINE_AA)
            return

        # Wrap items into lines from the tail (newest at bottom-right)
        max_w   = w - 20
        line_h  = LH_SMALL
        y_start = py + LH_TINY + 6
        max_lines = max(1, (h - LH_TINY - 8) // line_h)

        lines, truncated = _wrap_items_tail(items, max_w, FONT, FS_SMALL, max_lines)
        if truncated and lines:
            lines[0] = "..." + lines[0]

        y = y_start
        for line in lines:
            cv2.putText(canvas, line, (px + 10, y),
                        FONT, FS_SMALL, (140, 225, 150), FT_NORM, cv2.LINE_AA)
            y += line_h


# ── Helpers ────────────────────────────────────────────────────────────────

def _wrap_items_tail(items, max_w, font, scale, max_lines):
    """Build lines from the tail so newest item is bottom-right."""
    lines_rev = []
    current   = ""
    truncated = False
    for item in reversed(items):
        test = f"{item} {current}".strip()
        if cv2.getTextSize(test, font, scale, 1)[0][0] <= max_w:
            current = test
        else:
            lines_rev.append(current)
            if len(lines_rev) >= max_lines:
                truncated = True
                break
            current = item
    if not truncated and current:
        lines_rev.append(current)
    return list(reversed(lines_rev)), truncated
