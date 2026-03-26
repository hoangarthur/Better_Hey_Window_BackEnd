"""
StreamVerifier: Two-pass rescan confirmation for gesture detection.

Pipeline:
  FULL_SCAN     → collect ALL gestures scoring ≥ 80%, build sorted queue
  RESCAN PASS 1 → tight window (±0.5 s): confirm → fire, or escalate
  RESCAN PASS 2 → wide  window (±1.0 s): confirm → fire, or discard + next

Rules:
  • All candidates come from FULL_SCAN only.
    New gestures appearing during a rescan are NOT added to the queue.
  • Each candidate gets at most 2 passes.
    Pass-1 confirmed → fire immediately (no need for pass 2).
    Pass-1 failed    → run pass 2 on the same candidate.
    Pass-2 failed    → discard, move to next candidate in queue.
  • Queue is ordered by peak score (highest first).
  • No limit on number of candidates in the queue.
"""


class StreamVerifier:
    DEFAULT_CONFIRM_THRESHOLD = 0.40  # lowered: temperature=1.5 + many classes
    RESCAN_WINDOW_TIGHT  = 0.5   # seconds — pass 1 (tight)
    RESCAN_WINDOW_WIDE   = 1.0   # seconds — pass 2 (wide)

    def __init__(self, confirm_threshold: float | None = None):
        base = self.DEFAULT_CONFIRM_THRESHOLD if confirm_threshold is None else confirm_threshold
        self.confirm_threshold = self._clamp_threshold(base)
        self.reset()

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return max(0.05, min(0.95, float(value)))

    def set_confirm_threshold(self, value: float) -> None:
        self.confirm_threshold = self._clamp_threshold(value)

    # ─────────────────────────────────────────────
    def reset(self) -> None:
        self.phase: str = "FULL_SCAN"   # "FULL_SCAN" | "RESCAN" | "DONE"
        self.peaks: dict = {}           # {name: {"score": float, "frame_idx": int}}
        self._candidate_queue: list = []  # [(score, name, peak_info), ...] sorted desc
        self._queue_built: bool = False
        self.rescan_target: str | None = None
        self._current_pass: int = 0     # 1 = tight, 2 = wide
        self._needs_pass2: bool = False
        self.rescan_confirmed: bool = False
        self.rescan_peak_score: float = 0.0

    # ── FULL_SCAN phase ───────────────────────────
    def on_full_scan_frame(self, frame_idx: int, candidates: list) -> None:
        """Record the best score per gesture if it reaches the threshold."""
        for c in candidates:
            name = c.get("gesture_name", "")
            score = c.get("score", 0.0)
            if not name or name == "Unknown":
                continue
            if score >= self.confirm_threshold:
                if name not in self.peaks or score > self.peaks[name]["score"]:
                    self.peaks[name] = {"score": score, "frame_idx": frame_idx}

    def _build_queue(self) -> None:
        self._candidate_queue = sorted(
            [(v["score"], k, v) for k, v in self.peaks.items()],
            reverse=True,
        )
        self._queue_built = True

    # ── Rescan phase ──────────────────────────────
    @property
    def current_pass_num(self) -> int:
        return self._current_pass

    def get_rescan_window(self, total_frames: int, fps: float) -> tuple:
        """
        Returns (gesture_name, start_idx, end_idx) for the next window,
        or (None, 0, 0) when there is nothing left to try.

        Routing:
          _needs_pass2 = True  → wide window for the same candidate (pass 2)
          otherwise            → tight window for the next queued candidate (pass 1)
        """
        fps = max(fps, 1.0)

        # Route A: pass 2 for the candidate that just failed pass 1
        if self._needs_pass2 and self.rescan_target is not None:
            self._needs_pass2 = False
            self._current_pass = 2
            peak_info = self.peaks.get(self.rescan_target)
            if peak_info is None:
                return None, 0, 0
            peak_idx = peak_info["frame_idx"]
            hw = max(1, int(self.RESCAN_WINDOW_WIDE * fps))
            start = max(0, peak_idx - hw)
            end   = min(total_frames, peak_idx + hw)
            self.rescan_confirmed  = False
            self.rescan_peak_score = 0.0
            self.phase = "RESCAN"
            return self.rescan_target, start, end

        # Route B: pass 1 for the next candidate
        if not self._queue_built:
            self._build_queue()
        if not self._candidate_queue:
            return None, 0, 0

        _score, gname, peak_info = self._candidate_queue.pop(0)
        self.rescan_target     = gname
        self._current_pass     = 1
        self._needs_pass2      = False
        self.rescan_confirmed  = False
        self.rescan_peak_score = 0.0
        self.phase = "RESCAN"

        peak_idx = peak_info["frame_idx"]
        ht    = max(1, int(self.RESCAN_WINDOW_TIGHT * fps))
        start = max(0, peak_idx - ht)
        end   = min(total_frames, peak_idx + ht)
        return gname, start, end

    def on_rescan_frame(self, candidates: list) -> bool:
        """
        Called each scored frame during rescan playback.
        Only tracks the current target — new gestures are ignored.
        Returns True the moment the target is confirmed.
        """
        for c in candidates:
            if c.get("gesture_name") == self.rescan_target:
                score = c.get("score", 0.0)
                if score >= self.confirm_threshold:
                    self.rescan_confirmed = True
                    self.rescan_peak_score = max(self.rescan_peak_score, score)
                    return True
        return False

    def finish_rescan(self) -> tuple:
        """
        Called when the current rescan window finishes playback.
        Returns (confirmed: bool, gesture_name: str|None, score: float).

        • confirmed=True  → caller fires the gesture
        • confirmed=False → caller calls get_rescan_window() for the next attempt
        """
        if self.rescan_confirmed:
            self.phase = "DONE"
            return True, self.rescan_target, self.rescan_peak_score

        if self._current_pass == 1:
            # Pass 1 failed → request pass 2 for this same candidate
            self._needs_pass2 = True
            return False, self.rescan_target, 0.0

        # Pass 2 failed → discard this candidate, ready for next
        self.rescan_target = None
        self._current_pass = 0
        self._needs_pass2  = False
        self.phase = "FULL_SCAN"
        return False, None, 0.0

    def can_rescan(self) -> bool:
        """True if there are more rescan attempts still available."""
        return self._needs_pass2 or bool(self._candidate_queue)
