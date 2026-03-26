"""
Deprecated JSON template builder.

The project now uses an ML-only action pipeline (GestureTCN + MLMotionMatcher).
Generating JSON action templates is intentionally disabled.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve


class VideoTemplateBuilder:
    """Backward-compatible shim for old imports."""

    def __init__(self, min_hand_confidence: float = 0.35):
        self.min_hand_confidence = min_hand_confidence

    def build_action_template(
        self,
        video_path: str,
        gesture_id: str,
        gesture_name: str,
        output_path: Optional[str] = None,
        description: str = "",
        threshold: float = 0.55,
        required_hands: Optional[int] = None,
        sample_stride: int = 1,
        min_valid_frames: int = 16,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "JSON action template flow has been removed. "
            "Use ML pipeline: tools/ml_tuner.py -> models/gesture_model_best.pt."
        )


def download_video_to_temp(url: str) -> str:
    """Download a remote video URL to a temporary file and return local path."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported")

    suffix = Path(parsed.path).suffix.lower() or ".mp4"
    if len(suffix) > 8:
        suffix = ".mp4"

    tmp_dir = Path(tempfile.gettempdir()) / "betterheylab_video_import"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / f"import_{int(time.time())}{suffix}"

    urlretrieve(url, str(local_path))
    return str(local_path)
