"""
gesture_detector.py  —  Unified gesture detection pipeline
Separates detection into two channels:
  1. Character detection  (static hand shapes: A-Y, 0-9)
  2. Action detection     (dynamic hand motions: wave_hello, etc.)
"""

from typing import Dict, Optional, List, Tuple
from collections import deque

from pathlib import Path

from .character_matcher import CharacterMatcher
from core.ml_motion_matcher import MLMotionMatcher


class HandAnalysisCache:
    """
    Cache 3 analysis samples per hand before committing to a consensus result.
    Prevents single-frame noise from causing false detections.
    Uses pose-verified handedness to avoid left/right confusion.
    """
    REQUIRED_SAMPLES = 3

    def __init__(self):
        self._cache: Dict[str, list] = {"left": [], "right": []}
        self._final: Dict[str, Optional[dict]] = {"left": None, "right": None}

    def add_sample(self, side: str, match_result: dict) -> dict:
        """Add an analysis sample for the given hand side. Returns cache status."""
        if side not in ("left", "right"):
            return {"status": "invalid", "side": side, "samples": 0}

        self._cache[side].append(match_result)

        if len(self._cache[side]) >= self.REQUIRED_SAMPLES:
            self._final[side] = self._compute_consensus(side)
            return {
                "status": "ready",
                "side": side,
                "samples": len(self._cache[side]),
                "result": self._final[side],
            }

        return {
            "status": "caching",
            "side": side,
            "samples": len(self._cache[side]),
        }

    def _compute_consensus(self, side: str) -> Optional[dict]:
        """From REQUIRED_SAMPLES samples, pick the best consensus result.
        Consensus = gesture with most votes; tie-break by highest average score."""
        samples = self._cache[side]
        if not samples:
            return None

        groups: Dict[str, list] = {}
        for s in samples:
            key = s.get("gesture", "none")
            groups.setdefault(key, []).append(s)

        best_key = max(
            groups.keys(),
            key=lambda k: (
                len(groups[k]),
                sum(s.get("score", 0) for s in groups[k]) / len(groups[k]),
            ),
        )

        best_group = groups[best_key]
        peak_score = max(s.get("score", 0) for s in best_group)
        best_sample = max(best_group, key=lambda s: s.get("score", 0))

        return {
            "matched": best_sample.get("matched", False),
            "gesture": best_key,
            "gesture_name": best_sample.get("gesture_name", "Unknown"),
                "score": peak_score,
            "category": best_sample.get("category"),
            "consensus_count": len(best_group),
            "total_samples": len(samples),
        }

    def is_side_ready(self, side: str) -> bool:
        return self._final.get(side) is not None

    def get_final(self, side: str) -> Optional[dict]:
        return self._final.get(side)

    def reset_side(self, side: str):
        """Reset cache for one side (when hand is lost)."""
        if side in self._cache:
            self._cache[side] = []
            self._final[side] = None

    def reset_all(self):
        for side in ("left", "right"):
            self._cache[side] = []
            self._final[side] = None

    def get_combined_result(self) -> dict:
        return {
            "left": self._final.get("left"),
            "right": self._final.get("right"),
        }

    def get_debug_info(self) -> list:
        """Get debug entries for UI cache_history display."""
        lines = []
        for side in ("left", "right"):
            prefix = "L" if side == "left" else "R"
            for i, s in enumerate(self._cache[side]):
                name = s.get("gesture_name", "?")[:8]
                score = s.get("score", 0.0)
                pose = s.get("hand_pose", "?")
                spread = s.get("finger_spread", None)
                direction = s.get("direction", None)
                elbow = s.get("elbow_angle", None)
                row = {
                    "side": prefix,
                    "index": i+1,
                    "name": name,
                    "score": score,
                    "pose": pose,
                    "spread": spread,
                    "direction": direction,
                    "elbow_angle": elbow,
                }
                lines.append(row)
            final = self._final[side]
            if final:
                name = final.get("gesture_name", "?")[:8]
                score = final.get("score", 0.0)
                cnt = final.get("consensus_count", 0)
                total = final.get("total_samples", 0)
                row = {
                    "side": prefix,
                    "index": "final",
                    "name": name,
                    "score": score,
                    "pose": final.get("hand_pose", "?"),
                    "spread": final.get("finger_spread", None),
                    "direction": final.get("direction", None),
                    "elbow_angle": final.get("elbow_angle", None),
                    "consensus_count": cnt,
                    "total_samples": total,
                }
                lines.append(row)
        return lines
        return lines


class GestureDetector:
    """Unified detector with separated character and action pipelines."""
    
    def __init__(self, char_threshold: float = 0.60,
                 ml_model_path: str = "models/gesture_model_best.pt"):
        """
        Initialize both detection pipelines.

        Args:
            char_threshold: Minimum confidence score for character detection.
            ml_model_path:  Path to a trained GestureTCN checkpoint (.pt).
        """
        self.character_matcher = CharacterMatcher(threshold=char_threshold)
        self.char_threshold = char_threshold

        if not Path(ml_model_path).exists():
            raise FileNotFoundError(
                f"[GestureDetector] ML model không tìm thấy: '{ml_model_path}'\n"
                f"  Chạy 'python tools/ml_tuner.py' để train model trước."
            )
        self.motion_matcher = MLMotionMatcher(ml_model_path)
        print("[GestureDetector] Using ML motion matcher")
        self._char_history = {
            "left": deque(maxlen=5),
            "right": deque(maxlen=5)
        }
        self.hand_cache = HandAnalysisCache()
        
        # Last detected results (cached)
        self._last_char_result = None
        self._last_action_result = None
    
    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE 1: CHARACTER DETECTION (static)
    # ─────────────────────────────────────────────────────────────────────
    
    def detect_character(
        self,
        hand_landmarks: Optional[List[Tuple[float, float, float]]],
        side: str = "right",
        custom_threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect static hand characters (A-Y, 0-9).
        
        Args:
            hand_landmarks: 21-point hand landmarks
            side: "left" or "right"
            custom_threshold: Override default threshold
        
        Returns:
            {
                "detected": bool,
                "gesture_name": str,
                "score": float (0-1),
                "category": "character" | "number"
            }
        """
        result = {
            "detected": False,
            "gesture_name": None,
            "score": 0.0,
            "category": None,
            "channel": "character",
            "side": side,
            "cache_info": None,
        }
        
        if hand_landmarks is None:
            self.hand_cache.reset_side(side)
            self._char_history[side].clear()
            return result
        
        # Landmark-based matching (rotation/scale/translation invariant)
        threshold = custom_threshold or self.char_threshold
        match_result = self.character_matcher.match(hand_landmarks, threshold)

        # --- 3-sample cache mechanism ---
        # Each hand collects 3 independent analysis samples before committing.
        # Consensus from 3 samples filters out single-frame noise.
        cache_status = self.hand_cache.add_sample(side, match_result)
        result["cache_info"] = cache_status
        result["score"] = match_result.get("score", 0.0)
        result["gesture_name"] = match_result.get("gesture_name")
        result["category"] = match_result.get("category")

        if cache_status["status"] == "ready":
            consensus = cache_status["result"]
            # Reset cache for next analysis cycle
            self.hand_cache.reset_side(side)

            if consensus and consensus["score"] >= threshold:
                result["detected"] = True
                result["gesture_name"] = consensus["gesture_name"]
                result["score"] = consensus["score"]
                result["category"] = consensus["category"]
        
        self._last_char_result = result
        return result
    
    # ─────────────────────────────────────────────────────────────────────
    # PIPELINE 2: ACTION DETECTION (dynamic)
    # ─────────────────────────────────────────────────────────────────────
    
    def update_action_tracker(
        self,
        left_hand: Optional[List[Tuple[float, float, float]]] = None,
        right_hand: Optional[List[Tuple[float, float, float]]] = None,
        pose_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        snapshot: Optional[Dict] = None,
    ):
        """
        Update motion tracker with new hand landmarks and optional pose landmarks.

        Args:
            left_hand: 21-point left hand landmark list (pose-verified)
            right_hand: 21-point right hand landmark list (pose-verified)
            pose_landmarks: Optional pose landmarks for joint dependency checks
            snapshot: Optional holistic snapshot dict for metric-based v2 matching
        """
        self.motion_matcher.update(
            left_hand=left_hand,
            right_hand=right_hand,
            pose_landmarks=pose_landmarks,
            snapshot=snapshot,
        )
    
    def detect_action(self, batch_mode: bool = False) -> Dict:
        result = {
        "detected": False,
        "gesture": None,
        "gesture_name": None,
        "score": 0.0,
        "candidates": [],
        "category": "action",
        "channel": "action"
    }
        
        motion_result = self.motion_matcher.detect_motion(batch_mode=batch_mode)

        if motion_result["detected"]:

            result["detected"] = True
            result["gesture"] = motion_result.get("gesture")
            result["gesture_name"] = motion_result["gesture_name"]
            result["score"] = motion_result["score"]
            result["candidates"] = motion_result.get("candidates", [])
            result["debug"] = motion_result.get("debug", {})
        else:
            result["gesture"] = motion_result.get("gesture")
            result["gesture_name"] = motion_result.get("gesture_name")
            result["score"] = motion_result.get("score", 0.0)
            result["candidates"] = motion_result.get("candidates", [])
            result["debug"] = motion_result.get("debug", {})
        
        self._last_action_result = result
        return result
    
    # ─────────────────────────────────────────────────────────────────────
    # UNIFIED INTERFACE (optional)
    # ─────────────────────────────────────────────────────────────────────
    
    def detect_all(
        self,
        hand_landmarks: Optional[List[Tuple[float, float, float]]],
        custom_threshold: Optional[float] = None,
        action_priority: bool = True
    ) -> Dict:
        """
        Run both detection pipelines and return best result.
        
        Args:
            hand_landmarks: 21-point hand landmarks
            custom_threshold: Override character threshold
            action_priority: If True, prefer action over character when both detected
        
        Returns:
            Best result with "channel" field indicating which pipeline detected it
        """
        char_result = self.detect_character(hand_landmarks, custom_threshold)
        action_result = self._last_action_result  # Use cached from last detect_action call
        
        # Priority logic
        if action_priority and action_result and action_result["detected"]:
            return action_result
        elif char_result["detected"]:
            return char_result
        elif action_result and action_result["detected"]:
            return action_result
        else:
            # No detection - return char result (has better score details)
            return char_result
    
    # ─────────────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────────────
    
    def set_character_threshold(self, threshold: float):
        """Update character detection threshold (0.0 - 1.0)."""
        self.char_threshold = max(0.0, min(1.0, threshold))
        self.character_matcher.threshold = self.char_threshold
    
    def set_action_threshold(self, threshold: float):
        """Update action detection threshold (0.0 - 1.0)."""
        threshold = max(0.0, min(1.0, threshold))
        # Store in motion matcher's minimum score threshold if it exists
        if hasattr(self.motion_matcher, 'confidence_threshold'):
            self.motion_matcher.confidence_threshold = threshold

    def set_motion_min_frames(self, min_frames: int):
        """Update minimum required frames for motion matching."""
        if hasattr(self.motion_matcher, "set_min_required_frames"):
            self.motion_matcher.set_min_required_frames(min_frames)

    def set_motion_buffer_size(self, buffer_size: int):
        """Update motion buffer size at runtime."""
        if hasattr(self.motion_matcher, "set_buffer_size"):
            self.motion_matcher.set_buffer_size(buffer_size)

    def set_motion_boost(self, enabled: bool):
        """Enable/disable motion boost mode for normalized detection."""
        if hasattr(self.motion_matcher, "set_motion_boost"):
            self.motion_matcher.set_motion_boost(enabled)

    def set_gate_fail_factor(self, value: float):
        """Update runtime gate-fail penalty factor for action matching."""
        if hasattr(self.motion_matcher, "set_gate_fail_factor_override"):
            self.motion_matcher.set_gate_fail_factor_override(value)

    def reload_action_library(self):
        """Reload action templates from assets directory at runtime."""
        if hasattr(self.motion_matcher, "reload_action_library"):
            self.motion_matcher.reload_action_library()
    
    def get_last_results(self) -> Dict:
        """Get last detection results from both channels."""
        return {
            "character": self._last_char_result,
            "action": self._last_action_result
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded gestures."""
        return {
            "character_templates": len(self.character_matcher.templates),
            "action_templates": self.motion_matcher.gesture_manager.get_stats()["total_gestures"],
            "character_threshold": self.char_threshold
        }
