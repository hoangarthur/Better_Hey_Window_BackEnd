"""
ui/theme.py — shared visual constants for all panels.

Font scales are calibrated for panels ~500-900px wide on a 2K/4K monitor.
All panels import from here so changing one value updates everything.
"""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Font scales ────────────────────────────────────────────────────────────
FS_HERO   = 1.10   # large match % number
FS_TITLE  = 0.52   # section headers inside panels
FS_BODY   = 0.44   # main readable content
FS_SMALL  = 0.37   # secondary info
FS_TINY   = 0.30   # timestamps, dim metadata

FT_BOLD   = 2
FT_NORM   = 1

# ── Line heights ───────────────────────────────────────────────────────────
LH_TITLE  = 28
LH_BODY   = 22
LH_SMALL  = 18
LH_TINY   = 15

# ── Palette ────────────────────────────────────────────────────────────────
BG_PANEL      = (12, 12, 14)
BG_CARD       = (20, 20, 26)
BG_CARD_ALT   = (17, 19, 24)
BG_HDR        = (18, 18, 22)

COL_BORDER    = (44, 44, 56)
COL_BORDER_HI = (72, 72, 92)
COL_DIM       = (52, 52, 62)
COL_LABEL     = (105, 105, 120)
COL_TEXT      = (210, 210, 222)
COL_BRIGHT    = (235, 235, 245)

# Semantic colors
COL_ACTION    = ( 75, 215, 115)   # green  — action confirmed
COL_LETTER    = (105, 170, 255)   # blue   — character
COL_ACCENT        = (255, 205, 45)   # giữ nguyên cho các chỗ khác
COL_ACCENT_RESULT = (175, 155, 55)   # riêng cho TOOK line trong scan log
COL_PASS      = ( 55, 195,  85)   # bright green — rescan pass
COL_FAIL      = (195,  60,  55)   # red          — rescan fail
COL_WARN      = (220, 160,  45)   # amber        — partial/unsure
COL_STATUS    = (255, 220,  80)   # status line text
COL_CANDIDATE = (155, 215, 160)   # soft green   — candidate listing

# Card left-border accent
COL_CARD_OK   = ( 50, 195,  80)
COL_CARD_FAIL = (185,  55,  55)
COL_CARD_WARN = (185, 150,  40)

# ── Geometry ───────────────────────────────────────────────────────────────
PANEL_HDR_H = 26   # internal panel title bar height
CARD_PAD    = 10   # inner padding of session cards
CARD_GAP    =  7   # vertical gap between cards
SCROLLBAR_W =  5   # thin scrollbar on right edge


def draw_panel_header(canvas, x: int, y: int, w: int, text: str):
    """Standard panel title bar used by all panels."""
    cv2.rectangle(canvas, (x, y), (x + w, y + PANEL_HDR_H), BG_HDR, -1)
    cv2.putText(canvas, text, (x + 10, y + PANEL_HDR_H - 7),
                FONT, FS_SMALL, (195, 195, 205), FT_NORM, cv2.LINE_AA)
    cv2.line(canvas, (x, y + PANEL_HDR_H), (x + w, y + PANEL_HDR_H),
             COL_BORDER, 1)


def score_color(score: float):
    """Return color interpolated green→amber→red by score (1.0=green, 0=red)."""
    if score >= 0.75:
        return COL_ACTION
    elif score >= 0.50:
        return COL_WARN
    else:
        return COL_FAIL


def draw_score_bar(canvas, x: int, y: int, w: int, h: int,
                   score: float, color=None):
    """Filled progress bar with dark track."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (28, 28, 32), -1)
    if score > 0:
        fill_w = max(2, int(w * min(score, 1.0)))
        col = color or score_color(score)
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), col, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)
