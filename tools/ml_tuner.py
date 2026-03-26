"""
tools/ml_tuner.py
ML Gesture Training Tool
Output gesture_model_best.pt.
"""
from __future__ import annotations
import os
import re
import sys
import json
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
        


# ── Project imports ───────────────────────────────────────────────────────────
from core.detector import HolisticDetector, extract_snapshot
from core.group_tracker import GroupTracker
from core.ml_features import group_state_to_vector, FEATURE_DIM
from tools.train import GestureDataset, AugmentedDataset, train_from_arrays
# ── Optional yt_dlp ───────────────────────────────────────────────────────────
try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# ── Constants ─────────────────────────────────────────────────────────────────
TEXT_DATASET_DIR = os.path.join(PROJECT_ROOT, "textDataset")
LEGACY_CLIPS_DIR = os.path.join(PROJECT_ROOT, "clips")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model_best.pt")
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "gesture_dataset.npz")
NEW_LABEL_OPTION = "*Mới*"


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def parse_gesture_name(filename: str) -> str:
    """
    Parse gesture name from filename.
    Ví dụ:
        "wave_hello.mp4"          → "wave_hello"
        "Wave Hello (fast).mp4"   → "wave_hello"
        "p01_thank_you_001.mp4"   → "thank_you"
        "clip-again-repeat.avi"   → "again_repeat"
    """
    name = Path(filename).stem
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)                    # spaces/dashes → _
    name = re.sub(r"^p\d+_", "", name)                      # bỏ prefix p01_
    name = re.sub(r"_(normal|fast|slow|speed_\w+)$", "", name)  # bỏ speed suffix
    name = re.sub(r"_?\d{3,}$", "", name)                   # remove numeric suffix, e.g. _001
    name = re.sub(r"[^a-z0-9_]", "", name)                  # chỉ giữ a-z, 0-9, _
    name = re.sub(r"_+", "_", name).strip("_")              # collapse dấu _
    return name or "unknown"


def load_known_gestures() -> list[str]:
    """Danh sách gesture đã biết từ textDataset/, clips/ (legacy) + model checkpoint."""
    names: set[str] = set()

    for root in (Path(TEXT_DATASET_DIR), Path(LEGACY_CLIPS_DIR)):
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    names.add(d.name)

    try:
        import torch
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        for v in ckpt.get("label_map", {}).values():
            names.add(str(v))
    except Exception:
        pass

    return sorted(names)


# ═════════════════════════════════════════════════════════════════════════════
# TrainingChart — pure-tkinter live accuracy / loss chart
# ═════════════════════════════════════════════════════════════════════════════

class TrainingChart:
    W, H, PAD = 440, 210, 32

    def __init__(self, parent: tk.Widget):
        self.canvas = tk.Canvas(
            parent, width=self.W, height=self.H,
            bg="#1a1a2e", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.history: list[tuple[int, float, float, float]] = []
        self._total_epochs = 80
        self._draw_grid()

    # ── Public ────────────────────────────────────────────────────────────

    def reset(self, total_epochs: int = 80):
        self.history.clear()
        self._total_epochs = total_epochs
        self._draw_grid()

    def push(self, epoch: int, loss: float, train_acc: float, val_acc: float):
        self.history.append((epoch, loss, train_acc, val_acc))
        self._draw_grid()
        self._draw_curves()

    # ── Drawing ───────────────────────────────────────────────────────────

    def _x(self, epoch: int) -> float:
        W, P = self.W, self.PAD
        return P + (W - 2 * P) * epoch / max(self._total_epochs, 1)

    def _y(self, acc: float) -> float:
        H, P = self.H, self.PAD
        return P + (H - 2 * P) * (1.0 - max(0.0, min(1.0, acc)))

    def _draw_grid(self):
        c = self.canvas
        c.delete("all")
        W, H, P = self.W, self.H, self.PAD

        # Horizontal guidelines at 25 / 50 / 75 / 100 %
        for pct in (0, 25, 50, 75, 100):
            y = self._y(pct / 100)
            c.create_line(P, y, W - P, y, fill="#2a2a4a", dash=(3, 5))
            c.create_text(P - 5, y, text=f"{pct}%",
                          anchor="e", fill="#555", font=("Consolas", 7))

        # Axes
        c.create_line(P, P - 4, P, H - P + 4, fill="#444")
        c.create_line(P - 4, H - P, W - P + 4, H - P, fill="#444")

        # Epoch tick labels
        n = self._total_epochs
        for i in range(0, n + 1, max(1, n // 5)):
            x = self._x(i)
            c.create_line(x, H - P, x, H - P + 3, fill="#444")
            c.create_text(x, H - P + 10, text=str(i),
                          fill="#555", font=("Consolas", 7))

        # Y-axis label
        c.create_text(8, H // 2, text="acc", angle=90,
                      fill="#555", font=("Consolas", 8))

        # Legend
        c.create_line(W - 115, 14, W - 98, 14, fill="#4dd0e1", width=2)
        c.create_text(W - 95, 14, text="train", anchor="w",
                      fill="#4dd0e1", font=("Consolas", 8))
        c.create_line(W - 55, 14, W - 38, 14, fill="#a5d6a7", width=2)
        c.create_text(W - 35, 14, text="val", anchor="w",
                      fill="#a5d6a7", font=("Consolas", 8))

    def _draw_curves(self):
        if len(self.history) < 2:
            return
        c = self.canvas

        train_pts = [coord
                     for e, l, ta, va in self.history
                     for coord in (self._x(e), self._y(ta))]
        val_pts   = [coord
                     for e, l, ta, va in self.history
                     for coord in (self._x(e), self._y(va))]

        c.create_line(train_pts, fill="#4dd0e1", width=2, smooth=True)
        c.create_line(val_pts,   fill="#a5d6a7", width=2, smooth=True)

        # Mark best val point
        best = max(self.history, key=lambda r: r[3])
        bx, by = self._x(best[0]), self._y(best[3])
        c.create_oval(bx - 4, by - 4, bx + 4, by + 4,
                      fill="#ffcc02", outline="")
        c.create_text(bx, by - 10, text=f"{best[3]:.1%}",
                      fill="#ffcc02", font=("Consolas", 8))


# ═════════════════════════════════════════════════════════════════════════════
# TaskRow
# ═════════════════════════════════════════════════════════════════════════════

class TaskRow:
    """Represents a video-to-gesture labeling task in the UI."""

    def __init__(self, parent: tk.Widget, video_path: str,
                 is_yt: bool, display_name: str,
                 known_gestures: list[str], remove_cb):
        self.video_path   = video_path
        self.is_yt        = is_yt
        self.display_name = display_name

        self.frame = ttk.Frame(parent, relief="ridge", padding=4)

        # ── Tên file
        label = (display_name or os.path.basename(video_path))[:38]
        ttk.Label(self.frame, text=label, width=38, anchor="w",
                  font=("Consolas", 9)).pack(side="left", padx=(4, 2))

        # ── Gesture combo
        detected = parse_gesture_name(display_name or os.path.basename(video_path))
        self.gesture_var = tk.StringVar()

        options = known_gestures + (
            [detected] if detected not in known_gestures else []
        )
        options = [NEW_LABEL_OPTION] + options

        self.combo = ttk.Combobox(
            self.frame, textvariable=self.gesture_var,
            values=options, width=22, font=("Consolas", 9),
        )
        self.combo.pack(side="left", padx=4)

        # default selection: new if not recognized, else detected name
        if detected in known_gestures:
            self.gesture_var.set(detected)
        else:
            self.gesture_var.set(NEW_LABEL_OPTION)

        self.custom_var   = tk.StringVar(value=detected)
        self.custom_entry = ttk.Entry(self.frame, textvariable=self.custom_var,
                                      width=18, font=("Consolas", 9))

        def _on_combo(*_):
            if self.gesture_var.get() == NEW_LABEL_OPTION:
                self.custom_entry.pack(side="left", padx=2)
            else:
                self.custom_entry.pack_forget()

        self.combo.bind("<<ComboboxSelected>>", _on_combo)

        # if the auto-detected name is not in known gestures, show the custom entry by default
        if detected not in known_gestures:
            self.custom_entry.pack(side="left", padx=2)

        # ── Status
        self.status_var = tk.StringVar(value="Waiting")
        self._status_lbl = ttk.Label(
            self.frame, textvariable=self.status_var,
            width=22, anchor="w", font=("Consolas", 9), foreground="#888",
        )
        self._status_lbl.pack(side="left", padx=4)

        # ── Xóa
        ttk.Button(self.frame, text="✕", width=2,
                   command=lambda: remove_cb(self)).pack(side="right", padx=2)

    def gesture_name(self) -> str:
        v = self.gesture_var.get()
        if v == NEW_LABEL_OPTION:
            raw = self.custom_var.get().strip().lower()
        else:
            raw = v.strip().lower()
        raw = re.sub(r"[\s\-]+", "_", raw)
        raw = re.sub(r"[^a-z0-9_]", "", raw)
        return raw.strip("_") or "unknown"

    def set_status(self, text: str, color: str = "#888"):
        self.status_var.set(text)
        self._status_lbl.config(foreground=color)


# ═════════════════════════════════════════════════════════════════════════════
# MLTunerApp
# ═════════════════════════════════════════════════════════════════════════════

class MLTunerApp:
    def __init__(self, root: tk.Tk):
        self.root          = root
        self.root.title("ML Gesture Tuner")
        self.root.geometry("1060x720")
        self.root.configure(bg="#1e1e2e")

        self.tasks:          list[TaskRow] = []
        self.is_running:     bool          = False
        self.known_gestures: list[str]     = []

        self._apply_style()
        self._build_ui()
        self._refresh_gestures()

    # ── Style ─────────────────────────────────────────────────────────────

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        bg, fg, entry = "#1e1e2e", "#cdd6f4", "#313244"
        s.configure(".",           background=bg, foreground=fg,
                    fieldbackground=entry, font=("Consolas", 9))
        s.configure("TButton",     padding=5)
        s.configure("TFrame",      background=bg)
        s.configure("TLabelframe", background=bg, foreground="#89b4fa")
        s.configure("TLabelframe.Label", background=bg, foreground="#89b4fa",
                    font=("Consolas", 9, "bold"))
        s.configure("TLabel",      background=bg, foreground=fg)
        s.configure("TCombobox",   fieldbackground=entry, foreground=fg)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        # ─── Toolbar ────────────────────────────────────────────────────
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=10, pady=8)

        ttk.Button(bar, text="＋ Add Videos",
                   command=self.add_videos).pack(side="left", padx=4)

        if yt_dlp is not None:
            self.yt_var = tk.StringVar()
            ttk.Entry(bar, textvariable=self.yt_var,
                      width=32).pack(side="left", padx=2)
            ttk.Button(bar, text="＋ YouTube/Playlist",
                       command=self.add_youtube).pack(side="left", padx=4)

        ttk.Button(bar, text="Clear All",
                   command=self.clear_all).pack(side="right", padx=4)
        self.btn_start = ttk.Button(bar, text="▶  START TRAINING",
                                    command=self.start_training)
        self.btn_start.pack(side="right", padx=4)

        # ─── Main pane ──────────────────────────────────────────────────
        pane = ttk.Frame(self.root)
        pane.pack(fill="both", expand=True, padx=10, pady=2)
        pane.columnconfigure(0, weight=3)
        pane.columnconfigure(1, weight=2)
        pane.rowconfigure(0, weight=1)

        # Left: video queue
        q_frame = ttk.LabelFrame(pane, text="Video Queue  (gesture detect = file name)")
        q_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=2)

        q_canvas = tk.Canvas(q_frame, bg="#1e1e2e", highlightthickness=0)
        self._task_inner = ttk.Frame(q_canvas)
        q_scroll = ttk.Scrollbar(q_frame, orient="vertical",
                                  command=q_canvas.yview)
        q_canvas.configure(yscrollcommand=q_scroll.set)
        q_scroll.pack(side="right", fill="y")
        q_canvas.pack(side="left", fill="both", expand=True)
        q_canvas.create_window((0, 0), window=self._task_inner, anchor="nw")
        self._task_inner.bind(
            "<Configure>",
            lambda e: q_canvas.configure(scrollregion=q_canvas.bbox("all")),
        )

        # Right: training progress
        r_frame = ttk.Frame(pane)
        r_frame.grid(row=0, column=1, sticky="nsew", pady=2)

        chart_frame = ttk.LabelFrame(r_frame, text="Training Progress")
        chart_frame.pack(fill="both", expand=True)

        self.chart = TrainingChart(chart_frame)

        # Stats labels
        sf = ttk.Frame(chart_frame)
        sf.pack(fill="x", padx=8, pady=(0, 4))

        self._lbl_phase = ttk.Label(sf, text="Idle",
                                    font=("Consolas", 8, "italic"),
                                    foreground="#6c7086")
        self._lbl_phase.pack(anchor="w")

        self._progress = ttk.Progressbar(chart_frame, mode="determinate",
                                          maximum=100)
        self._progress.pack(fill="x", padx=8, pady=(0, 4))

        stats = ttk.Frame(chart_frame)
        stats.pack(fill="x", padx=8, pady=2)

        self._s_epoch = self._stat_lbl(stats, "Epoch",    "#cdd6f4")
        self._s_train = self._stat_lbl(stats, "Train",    "#4dd0e1")
        self._s_val   = self._stat_lbl(stats, "Val",      "#a5d6a7")
        self._s_best  = self._stat_lbl(stats, "Best val", "#ffcc02")
        self._s_loss  = self._stat_lbl(stats, "Loss",     "#f38ba8")

        # ─── Log ────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="x", padx=10, pady=(2, 8))

        self._log = tk.Text(
            log_frame, height=8, wrap="word",
            bg="#11111b", fg="#a6e3a1",
            font=("Consolas", 9), state="disabled",
        )
        log_sb = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self._log.pack(fill="both", padx=4, pady=4)

    def _stat_lbl(self, parent, prefix: str, color: str) -> ttk.Label:
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=1)
        ttk.Label(f, text=f"{prefix}:", width=9, foreground="#6c7086").pack(side="left")
        lbl = ttk.Label(f, text="—", foreground=color,
                        font=("Consolas", 9, "bold"))
        lbl.pack(side="left")
        return lbl

    # ── Gestures ──────────────────────────────────────────────────────────

    def _refresh_gestures(self):
        self.known_gestures = load_known_gestures()
        count = len(self.known_gestures)
        self.log(f"Known {count} gesture: "
                 + (", ".join(self.known_gestures) if count else "(New)"))

    # ── Add tasks ─────────────────────────────────────────────────────────

    def add_videos(self):
        files = filedialog.askopenfilenames(
            title="Add video gesture",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.webm")],
        )
        for f in files:
            self._add_task(f, is_yt=False)

    def add_youtube(self):
        url = self.yt_var.get().strip()
        if not url:
            return
        # if it's a playlist URL, convert to playlist format to get all videos
        params = parse_qs(urlparse(url).query)
        if "list" in params:
            url = f"https://www.youtube.com/playlist?list={params['list'][0]}"
        self.log(f"fetching video from YouTube: {url}")
        self.btn_start.config(state="disabled")
        threading.Thread(target=self._yt_info, args=(url,), daemon=True).start()

    def _yt_info(self, url: str):
        try:
            with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
            if "entries" in info:
                for e in info["entries"]:
                    vid_id  = e.get("id") or ""
                    vid_url = e.get("url") or ""
                    # flat extraction might give either full URL or just ID, handle both
                    if not vid_url.startswith("http"):
                        vid_url = f"https://www.youtube.com/watch?v={vid_id or vid_url}"
                    self.root.after(0, self._add_task,
                                    vid_url, True,
                                    e.get("title", "yt_video"))
                self.log(f"Playlist: {info.get('title')} — {len(info['entries'])} video")
            else:
                self.root.after(
                    0, self._add_task,
                    info.get("webpage_url", url), True,
                    info.get("title", "yt_video"),
                )
        except Exception as ex:
            self.root.after(0, self.log, f"[YT Error] {ex}")
        finally:
            self.root.after(0, lambda: self.btn_start.config(state="normal"))
            self.root.after(0, lambda: self.yt_var.set(""))

    def _add_task(self, path: str, is_yt: bool = False, display: str = ""):
        task = TaskRow(
            self._task_inner, path, is_yt,
            display or os.path.basename(path),
            self.known_gestures, self.remove_task,
        )
        task.frame.pack(fill="x", padx=4, pady=2)
        self.tasks.append(task)

    def remove_task(self, task: TaskRow):
        task.frame.destroy()
        if task in self.tasks:
            self.tasks.remove(task)

    def clear_all(self):
        for t in list(self.tasks):
            t.frame.destroy()
        self.tasks.clear()

    # ── Start training ────────────────────────────────────────────────────

    def start_training(self):
        if self.is_running:
            return
        if not self.tasks:
            messagebox.showwarning("Null", "add at least one video file")
            return
        for t in self.tasks:
            if not t.gesture_name() or t.gesture_name() == "unknown":
                messagebox.showwarning(
                    "need lable",
                    f"Video '{os.path.basename(t.video_path)}' need gesture name.",
                )
                return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.chart.reset()
        self._progress["value"] = 0
        threading.Thread(target=self._pipeline, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════════
    # Background pipeline
    # ═══════════════════════════════════════════════════════════════════════

    def _pipeline(self):
        self._phase("Phase 1/3 — trying to get input video…")
        self.log("=" * 55)
        self.log("▶ TRAINING PIPELINE")
        self.log("=" * 55)

        # ── Phase 1: extract features ──────────────────────────────────
        detector   = HolisticDetector(skip_frames=1, min_hand_confidence=0.25)
        new_seqs:  list[np.ndarray] = []
        new_labels: list[str]       = []

        for i, task in enumerate(self.tasks):
            g_name = task.gesture_name()
            self._progress_ui(int(i / len(self.tasks) * 30))
            self.root.after(0, task.set_status, "video is processing…", "#f9e2af")
            self.log(f"\n[{i+1}/{len(self.tasks)}] {os.path.basename(task.video_path)}"
                     f"  →  gesture: '{g_name}'")

            try:
                # Tải video (YT → download trước)
                vpath = self._resolve_video(task)
                if not vpath:
                    self.root.after(0, task.set_status, "Skip: load failed", "#f38ba8")
                    continue

                archived_path = self._archive_video_to_text_dataset(vpath, g_name)
                if not archived_path:
                    self.root.after(0, task.set_status, "Skip: save failed", "#f38ba8")
                    continue

                seq = self._extract_video(archived_path, detector)
                if seq is None:
                    self.root.after(0, task.set_status, "Skip: no hands detected", "#f38ba8")
                    self.log("  → SKIP: MediaPipe can not detect clear hand pose in this video.")
                    continue

                new_seqs.append(seq)
                new_labels.append(g_name)
                n = len(seq)
                self.root.after(0, task.set_status, f"✓ {n} frames", "#a6e3a1")
                self.log(f"  → OK: {n} frames (source: textDataset/{g_name}/)")

            except Exception as ex:
                self.root.after(0, task.set_status, "Skip: processing failed", "#f38ba8")
                self.log(f"  → ERROR: {ex}")

        detector.close()

        if not new_seqs:
            self.log("\n[ERROR] no video processed successfully. Please check the input files and try again.")
            self._done()
            return

        # ── Phase 2: build dataset ─────────────────────────────────────
        self._phase("Phase 2/3 — building dataset…")
        self._progress_ui(32)

        try:
            X, y, label_map = self._build_dataset(new_seqs, new_labels)
        except Exception as ex:
            self.log(f"\n[ERROR] building dataset failed: {ex}")
            self._done()
            return

        self._progress_ui(40)

        # ── Phase 3: train ─────────────────────────────────────────────
        self._phase("Phase 3/3 — building model…")
        self.log(f"\ntraining: {len(np.unique(y))} classes, "
                 f"{len(y)} samples, dim={X.shape[2]}")

        epochs     = 80
        best_acc   = 0.0
        history:   list[tuple] = []

        def on_epoch(epoch, total, loss, train_acc, val_acc) -> bool:
            nonlocal best_acc
            history.append((epoch, loss, train_acc, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
            pct = 40 + int(epoch / total * 60)
            # update UI — thread-safe
            self.root.after(0, self._update_training_ui,
                            epoch, total, loss, train_acc, val_acc,
                            best_acc, list(history), pct)
            if epoch % 10 == 0 or epoch == 1:
                star = " ←best" if val_acc == best_acc else ""
                self.log(f"  Epoch {epoch:3d}/{total}  "
                         f"loss={loss:.4f}  "
                         f"train={train_acc:.1%}  "
                         f"val={val_acc:.1%}{star}")
            return False # return True to stop training early

        try:
            best_acc = train_from_arrays(
                X, y, label_map,
                output_dir=os.path.dirname(MODEL_PATH),
                epochs=epochs,
                on_epoch=on_epoch,
            )
        except Exception as ex:
            self.log(f"\n[ERROR] Training failed: {ex}")
            self._done()
            return

        self.log(f"\n{'='*55}")
        self.log(f"✓  Training completed  —  Best val acc: {best_acc:.1%}")
        self.log(f"✓  Model saved to: {MODEL_PATH}")
        self.log(f"{'='*55}")
        self._done()

    # ── Video helpers ──────────────────────────────────────────────────────

    def _resolve_video(self, task: TaskRow) -> Optional[str]:
        """return local video path, or None if failed. For YouTube, download first."""
        if not task.is_yt:
            return task.video_path if os.path.exists(task.video_path) else None
        if yt_dlp is None:
            self.log("  → yt-dlp need to be installed (pip install yt-dlp)")
            return None
        try:
            import tempfile
            tmp = tempfile.mkdtemp()
            opts = {
                "format": "mp4/bestvideo[ext=mp4]",
                "outtmpl": os.path.join(tmp, "%(title)s.%(ext)s"),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(task.video_path, download=True)
                return ydl.prepare_filename(info)
        except Exception as ex:
            self.log(f"  → YT download error: {ex}")
            return None

    def _archive_video_to_text_dataset(self, src_video_path: str, gesture_name: str) -> Optional[str]:
        """
        Copy video into textDataset/<gesture_name>/ and return archived path.
        Uses unique filename when collision occurs.
        """
        try:
            src = Path(src_video_path).resolve()
            if not src.exists():
                self.log(f"  → Source video not found: {src_video_path}")
                return None

            dst_dir = Path(TEXT_DATASET_DIR) / gesture_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            target = dst_dir / src.name
            if target.exists():
                try:
                    if src.samefile(target):
                        self.log(f"  → Existing: {target}")
                        return str(target)
                except OSError:
                    pass
                if src.stat().st_size == target.stat().st_size:
                    self.log(f"  → Reuse existing: {target}")
                    return str(target)

                stem = target.stem
                suffix = target.suffix
                idx = 1
                while True:
                    candidate = dst_dir / f"{stem}_{idx:03d}{suffix}"
                    if not candidate.exists():
                        target = candidate
                        break
                    idx += 1

            shutil.copy2(src, target)
            self.log(f"  → Stored: {target}")
            return str(target)
        except Exception as ex:
            self.log(f"  → Save video failed: {ex}")
            return None

    @staticmethod
    def _extract_video(video_path: str,
                       detector: HolisticDetector) -> Optional[np.ndarray]:
        """Extract feature sequence 110-dim from a video file."""
        tracker = GroupTracker()
        cap     = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []

        while len(frames) < 120:
            ret, frame = cap.read()
            if not ret:
                break
            mp = detector.process(frame)
            if not (mp.get("left_hand_landmarks") or mp.get("right_hand_landmarks")):
                if len(frames) > 5:
                    break
                continue
            snap   = extract_snapshot(mp)
            states = tracker.compute(snap)
            frames.append(group_state_to_vector(states, snap))

        cap.release()
        return np.stack(frames) if len(frames) >= 8 else None

    # ── Dataset builder ───────────────────────────────────────────────────

    def _build_dataset(
        self,
        new_seqs:   list[np.ndarray],
        new_labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Merge new sequences with the existing dataset (if compatible).
        Return (X, y, label_map).
        """
        all_seqs   = list(new_seqs)
        all_labels = list(new_labels)

        # Merge với dataset cũ nếu tồn tại + đúng dim
        data_file = Path(DATA_PATH)
        if data_file.exists():
            try:
                saved  = np.load(str(data_file))
                X_old  = saved["X"]
                y_old  = saved["y"]
                lbl_fp = str(data_file).replace(".npz", "_labels.json")
                with open(lbl_fp, encoding="utf-8") as f:
                    old_map = {int(k): v for k, v in json.load(f).items()}

                if X_old.shape[2] != FEATURE_DIM:
                    self.log(f"[WARN] Dataset old dim={X_old.shape[2]} ≠ "
                             f"FEATURE_DIM={FEATURE_DIM} → skip old data")
                else:
                    for xi, yi in zip(X_old, y_old):
                        name = old_map.get(int(yi))
                        if name:
                            all_seqs.append(xi)
                            all_labels.append(name)
                    self.log(f"load {len(X_old)} old sample from dataset")
            except Exception as ex:
                self.log(f"[WARN] Could not load old dataset: {ex}")

        # Xây label map từ tất cả tên unique (sắp xếp để ổn định)
        unique_names  = sorted(set(all_labels))
        label_map     = {i: n for i, n in enumerate(unique_names)}
        label_map_inv = {n: i for i, n in label_map.items()}

        y_list = [label_map_inv[n] for n in all_labels]

        max_len = min(max(len(s) for s in all_seqs), 90)
        padded  = np.stack([_pad_seq(s, max_len) for s in all_seqs])
        y_arr   = np.array(y_list, dtype=np.int64)

        # Lưu dataset
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        np.savez(DATA_PATH, X=padded, y=y_arr,
                 lengths=np.full(len(y_arr), max_len, dtype=np.int64),
                 max_len=max_len)
        json_map = {str(k): v for k, v in label_map.items()}
        with open(DATA_PATH.replace(".npz", "_labels.json"), "w",
                  encoding="utf-8") as f:
            json.dump(json_map, f, indent=2, ensure_ascii=False)

        self.log(f"\nDataset: shape={padded.shape}, {len(label_map)} classes")
        from collections import Counter
        for idx, cnt in sorted(Counter(y_list).items()):
            flag = "  ⚠ ít" if cnt < 15 else ""
            self.log(f"  [{idx:2d}] {label_map[idx]:25s}: {cnt} mẫu{flag}")

        return padded, y_arr, label_map

    # ── Thread-safe UI updaters ────────────────────────────────────────────

    def _update_training_ui(self, epoch, total, loss, train_acc, val_acc,
                             best_acc, history, pct):
        self._s_epoch.config(text=f"{epoch} / {total}")
        self._s_train.config(text=f"{train_acc:.1%}")
        self._s_val.config(  text=f"{val_acc:.1%}")
        self._s_best.config( text=f"{best_acc:.1%}")
        self._s_loss.config( text=f"{loss:.4f}")
        self._progress["value"] = pct
        self.chart.push(epoch, loss, train_acc, val_acc)

    def _phase(self, text: str):
        self.root.after(0, lambda t=text: self._lbl_phase.config(text=t))
        self.log(f"\n── {text}")

    def _progress_ui(self, pct: int):
        self.root.after(0, lambda p=pct: self._progress.config(value=p))

    def _done(self):
        self.root.after(0, self._done_ui)

    def _done_ui(self):
        self.is_running = False
        self.btn_start.config(state="normal")
        self._lbl_phase.config(text="Completed ✓")
        self._progress["value"] = 100
        self._refresh_gestures()

    def log(self, msg: str):
        def _do():
            self._log.config(state="normal")
            self._log.insert(tk.END, msg + "\n")
            self._log.see(tk.END)
            self._log.config(state="disabled")
        self.root.after(0, _do)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _pad_seq(seq: np.ndarray, target: int) -> np.ndarray:
    T = len(seq)
    if T >= target:
        s = (T - target) // 2
        return seq[s:s + target]
    return np.vstack([
        np.zeros((target - T, seq.shape[1]), dtype=np.float32),
        seq,
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    root = tk.Tk()
    app  = MLTunerApp(root)
    root.mainloop()
