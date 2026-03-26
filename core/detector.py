"""
core/detector.py
Keypoint detector — uses MediaPipe Tasks API (v0.10+).
Falls back to animated DEMO skeleton when model .task files are absent.

Model files (place in assets/models/):
  pose_landmarker_full.task
  hand_landmarker.task
  face_landmarker.task

Download: https://developers.google.com/mediapipe/solutions/vision
"""
import importlib.util, sys, math
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).parent.parent

COL_FACE = (0, 220, 255)
COL_HAND = (0, 255, 140)
COL_POSE = (255, 180, 30)
DOT_R    = 2
LINE_T   = 1
LABEL_FONT_SCALE = 0.28
LABEL_THICKNESS = 1
SHOW_LANDMARK_LABELS = True

# Find MediaPipe dynamically across platforms
def _find_mediapipe_path():
    """Dynamically locate MediaPipe tasks modules."""
    try:
        # Try direct import first (more reliable)
        from mediapipe.tasks.python import vision, core
        return True, vision, core
    except ImportError:
        pass
    return False, None, None

_MP_FOUND, _MP_VISION, _MP_CORE = _find_mediapipe_path()


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class HolisticDetector:
    def __init__(self, skip_frames=1, min_hand_confidence=0.25):
        """
        skip_frames: process every Nth frame (1=all, 2=every 2nd, 3=every 3rd)
        min_hand_confidence: filter hands below this confidence (0-1)
        """
        self._pose = self._hand = self._face = None
        self._demo = False
        self._demo_t = 0.0
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.min_hand_confidence = min_hand_confidence
        self._last_results = _empty()
        self._try_init()

    def _try_init(self):
        models = {
            "pose": ROOT / "assets/models/pose_landmarker_full.task",
            "hand": ROOT / "assets/models/hand_landmarker.task",
            "face": ROOT / "assets/models/face_landmarker.task",
        }
        missing = [k for k, p in models.items() if not p.exists()]
        if missing:
            print(f"[detector] Missing models {missing} → DEMO mode (animated skeleton)")
            print("  To use real detection, download .task files to assets/models/")
            self._demo = True
            return
        
        if not _MP_FOUND:
            print("[detector] MediaPipe library not found → DEMO mode")
            print("  Install: pip install mediapipe")
            self._demo = True
            return
        
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core import base_options as bo_module
            
            # Create base options with model paths
            base_options_pose = bo_module.BaseOptions(
                model_asset_path=str(models["pose"]))
            base_options_hand = bo_module.BaseOptions(
                model_asset_path=str(models["hand"]))
            base_options_face = bo_module.BaseOptions(
                model_asset_path=str(models["face"]))
            
            # Create detectors with options
            self._pose = _MP_VISION.PoseLandmarker.create_from_options(
                _MP_VISION.PoseLandmarkerOptions(
                    base_options=base_options_pose))
            self._hand = _MP_VISION.HandLandmarker.create_from_options(
                _MP_VISION.HandLandmarkerOptions(
                    base_options=base_options_hand,
                    num_hands=2,
                    min_hand_detection_confidence=0.1,
                    min_hand_presence_confidence=0.1,
                    min_tracking_confidence=0.1))  # Enable dual-hand detection and lower confidence for better recall
            self._face = _MP_VISION.FaceLandmarker.create_from_options(
                _MP_VISION.FaceLandmarkerOptions(
                    base_options=base_options_face))
            
            print("[detector] MediaPipe Tasks ready (offline mode)")
            print("[detector] Dual-hand detection enabled")
        except Exception as e:
            print(f"[detector] Init error ({e}) → DEMO mode")
            import traceback
            traceback.print_exc()
            self._demo = True

    def process(self, bgr: np.ndarray) -> dict:
        """
        Process frame with skip_frames logic.
        Returns last valid results on skipped frames.
        """
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.skip_frames != 0:
            return self._last_results
        
        if self._demo:
            import time; self._demo_t = time.time()
            self._last_results = _demo_results(self._demo_t)
            return self._last_results
        
        try:
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pr = self._pose.detect(img) if self._pose else None
            hr = self._hand.detect(img) if self._hand else None
            fr = self._face.detect(img) if self._face else None

            pose_lms = [(lm.x, lm.y, lm.z) for lm in pr.pose_landmarks[0]] \
                       if pr and pr.pose_landmarks else None
            
            lh, rh = None, None
            hands_detected = 0
            if hr and hr.hand_landmarks:
                valid_hands = []
                for i, cats in enumerate(hr.handedness):
                    if i >= len(hr.hand_landmarks):
                        break
                    if cats[0].score < self.min_hand_confidence:
                        continue
                    side = cats[0].category_name.strip().lower()
                    lms = [(lm.x, lm.y, lm.z) for lm in hr.hand_landmarks[i]]
                    valid_hands.append({"lms": lms, "side": side})
                    hands_detected += 1
                
                # Match Hand to Pose Wrists for perfectly stable Left/Right mapping regardless of MediaPipe Hand predictions 
                if pose_lms and len(pose_lms) >= 17:
                    lw = pose_lms[15] # Left wrist
                    rw = pose_lms[16] # Right wrist
                    for hand in valid_hands:
                        # calculate distance from hand wrist to pose wrists
                        hx, hy = hand["lms"][0][:2]
                        d_left = math.hypot(hx - lw[0], hy - lw[1])
                        d_right = math.hypot(hx - rw[0], hy - rw[1])
                        
                        if d_left < d_right:
                            if lh is None or d_left < math.hypot(lh[0][0]-lw[0], lh[0][1]-lw[1]):
                                lh = hand["lms"]
                        else:
                            if rh is None or d_right < math.hypot(rh[0][0]-rw[0], rh[0][1]-rw[1]):
                                rh = hand["lms"]
                # Không có pose wrist → không thể xác minh tay trái/phải → bỏ qua
                # Tránh nhận diện nhầm khi chỉ dựa vào MediaPipe handedness label
                        
            # Đã gỡ bỏ Fallback tạo 21 điểm giả (làm hỏng group_tracker hand pose)
            # Thay vào đó, việc fallback tọa độ cổ tay sẽ được xử lý sạch sẽ ở Data Layer (group_tracker.py)
            
            if self.frame_count % 30 == 1:
                total_hands_val = len(hr.hand_landmarks) if hr and hr.hand_landmarks else 0
                print(f"[hand] MediaPipe detected {total_hands_val} hand(s) raw, "
                      f"{hands_detected} passed confidence filter → "
                      f"left_hand={'YES' if lh else 'NO'}, right_hand={'YES' if rh else 'NO'}")

            face_lms = [(lm.x, lm.y, lm.z) for lm in fr.face_landmarks[0]] \
                       if fr and fr.face_landmarks else None
            
            result = {"pose_landmarks": pose_lms, "left_hand_landmarks": lh,
                    "right_hand_landmarks": rh, "face_landmarks": face_lms}
            self._last_results = result
            return result
        except Exception as e:
            print(f"[detector] {e}")
            return self._last_results

    def close(self):
        for d in [self._pose, self._hand, self._face]:
            if d:
                try: d.close()
                except: pass

    def set_performance(self, skip_frames: int, min_hand_confidence: float):
        """Adjust performance settings on-the-fly."""
        self.skip_frames = max(1, skip_frames)
        self.min_hand_confidence = max(0.0, min(1.0, min_hand_confidence))
        self.frame_count = 0
        print(f"[detector] Performance: skip_frames={self.skip_frames}, "
              f"min_hand_confidence={self.min_hand_confidence:.2f}")


# ── Draw helpers ───────────────────────────────────────────────────────────
POSE_CONN = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
]
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
# Face connections: only key contours (face outline, eyes, mouth)
# Indices for MediaPipe face landmarks
FACE_CONN = [
    # Face outline (simplified)
    (10,338),(338,297),(297,332),(332,284),(284,251),
    (251,389),(389,356),(356,454),(454,323),(323,361),
    (361,288),(288,397),(397,365),(365,379),(379,378),
    (378,400),(400,377),(377,152),(152,148),(148,176),
    (176,149),(149,150),(150,136),(136,172),(172,58),
    (58,132),(132,93),(93,234),(234,127),(127,162),
    # Right eye
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),
    # Left eye  
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),
    # Mouth (simplified)
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),
    (405,321),(321,375),(375,291),(291,61),
]

def _pts(lms, rect, mirror=True):
    x,y,w,h = rect
    return [(int((1.0 - lm[0] if mirror else lm[0]) * w) + x, int(lm[1] * h) + y) for lm in lms]


def _draw_label(canvas, point, text, color, dx=4, dy=-4):
    tx, ty = point[0] + dx, point[1] + dy
    cv2.putText(
        canvas,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        color,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )

def draw_face_zone(canvas, res, rect):
    """Draw face with key contours only (not all 468 landmarks)."""
    lms = res.get("face_landmarks")
    if not lms: return
    pts = _pts(lms, rect)
    
    # Draw face contours (connections)
    for a,b in FACE_CONN:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas, pts[a], pts[b], COL_FACE, LINE_T)
    
    # Draw dots only at connection endpoints to reduce clutter
    drawn = set()
    for a,b in FACE_CONN:
        for idx in (a,b):
            if idx < len(pts) and idx not in drawn:
                cv2.circle(canvas, pts[idx], DOT_R, COL_FACE, -1)
                drawn.add(idx)

    if SHOW_LANDMARK_LABELS:
        # Compact symbolic labels for key face points.
        key_labels = {
            1: "F1",
            33: "F2",
            263: "F3",
            61: "F4",
            291: "F5",
        }
        for idx, label in key_labels.items():
            if idx < len(pts):
                _draw_label(canvas, pts[idx], label, COL_FACE)

def draw_hand_zone(canvas, res, rect):
    for key in ("left_hand_landmarks","right_hand_landmarks"):
        lms = res.get(key)
        if not lms: continue
        pts = _pts(lms, rect)
        for a,b in HAND_CONN: cv2.line(canvas, pts[a], pts[b], COL_HAND, LINE_T)
        for pt in pts: cv2.circle(canvas, pt, DOT_R+1, COL_HAND, -1)

def draw_left_hand(canvas, res, rect):
    """Draw only left hand landmarks."""
    lms = res.get("left_hand_landmarks")
    if not lms: return
    pts = _pts(lms, rect)
    for a,b in HAND_CONN: cv2.line(canvas, pts[a], pts[b], COL_HAND, LINE_T)
    for pt in pts: cv2.circle(canvas, pt, DOT_R+1, COL_HAND, -1)
    if SHOW_LANDMARK_LABELS:
        for i, pt in enumerate(pts):
            if i < len(HAND_NAMES):
                _draw_label(canvas, pt, f"A{i+1}", COL_HAND)

def draw_right_hand(canvas, res, rect):
    """Draw only right hand landmarks."""
    lms = res.get("right_hand_landmarks")
    if not lms: return
    pts = _pts(lms, rect)
    for a,b in HAND_CONN: cv2.line(canvas, pts[a], pts[b], COL_HAND, LINE_T)
    for pt in pts: cv2.circle(canvas, pt, DOT_R+1, COL_HAND, -1)
    if SHOW_LANDMARK_LABELS:
        for i, pt in enumerate(pts):
            if i < len(HAND_NAMES):
                _draw_label(canvas, pt, f"A{i+1}", COL_HAND)

def draw_pose_zone(canvas, res, rect):
    lms = res.get("pose_landmarks")
    if not lms: return
    pts = _pts(lms, rect)
    for a,b in POSE_CONN:
        if a<len(pts) and b<len(pts): cv2.line(canvas, pts[a], pts[b], COL_POSE, LINE_T)
    for pt in pts: cv2.circle(canvas, pt, DOT_R, COL_POSE, -1)
    if SHOW_LANDMARK_LABELS:
        for i, pt in enumerate(pts):
            if i < len(POSE_NAMES):
                _draw_label(canvas, pt, f"C{i+1}", COL_POSE)


# ── Snapshot ───────────────────────────────────────────────────────────────
POSE_NAMES = ["nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear",
    "mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow",
    "right_elbow","left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"]
HAND_NAMES = ["wrist","thumb_cmc","thumb_mcp","thumb_ip","thumb_tip",
    "index_finger_mcp","index_finger_pip","index_finger_dip","index_finger_tip",
    "middle_finger_mcp","middle_finger_pip","middle_finger_dip","middle_finger_tip",
    "ring_finger_mcp","ring_finger_pip","ring_finger_dip","ring_finger_tip",
    "pinky_mcp","pinky_pip","pinky_dip","pinky_tip"]

def extract_snapshot(res: dict) -> dict:
    def _make(lms, names):
        if not lms: return None
        return {names[i]: tuple(lm[:3]) for i,lm in enumerate(lms) if i<len(names)}
    snapshot = {
        "pose":       _make(res.get("pose_landmarks"),       POSE_NAMES),
        "left_hand":  _make(res.get("left_hand_landmarks"),  HAND_NAMES),
        "right_hand": _make(res.get("right_hand_landmarks"), HAND_NAMES),
        "face":       {str(i): tuple(lm[:3]) for i,lm in enumerate(res["face_landmarks"])}
                      if res.get("face_landmarks") else None,
    }
    return snapshot


# ── Demo skeleton ──────────────────────────────────────────────────────────
def _empty():
    return {"pose_landmarks":None,"left_hand_landmarks":None,
            "right_hand_landmarks":None,"face_landmarks":None}

def _demo_results(t: float) -> dict:
    w = math.sin(t * 2.5) * 0.08
    pose = [
        (0.50,0.15,0),(0.49,0.13,0),(0.48,0.13,0),(0.47,0.13,0),
        (0.51,0.13,0),(0.52,0.13,0),(0.53,0.13,0),(0.46,0.15,0),(0.54,0.15,0),
        (0.49,0.17,0),(0.51,0.17,0),
        (0.42,0.32,0),(0.58,0.32,0),(0.38,0.45,0),(0.62,0.45,0),
        (0.36,0.55+w,0),(0.64,0.35-w,0),
        (0.35,0.57,0),(0.65,0.33,0),(0.35,0.57,0),(0.65,0.33,0),
        (0.36,0.56,0),(0.64,0.34,0),
        (0.44,0.60,0),(0.56,0.60,0),(0.43,0.75,0),(0.57,0.75,0),
        (0.43,0.90,0),(0.57,0.90,0),(0.43,0.92,0),(0.57,0.92,0),
        (0.44,0.93,0),(0.56,0.93,0),
    ]
    rx, ry = 0.64, 0.35 - w
    rh = [
        (rx,ry,0),(rx+0.02,ry-0.02,0),(rx+0.03,ry-0.04,0),
        (rx+0.04,ry-0.05,0),(rx+0.05,ry-0.06,0),
        (rx+0.02,ry-0.04,0),(rx+0.02,ry-0.06,0),(rx+0.02,ry-0.08,0),(rx+0.02,ry-0.09,0),
        (rx+0.00,ry-0.04,0),(rx+0.00,ry-0.06,0),(rx+0.00,ry-0.08,0),(rx+0.00,ry-0.09,0),
        (rx-0.02,ry-0.04,0),(rx-0.02,ry-0.06,0),(rx-0.02,ry-0.07,0),(rx-0.02,ry-0.08,0),
        (rx-0.04,ry-0.03,0),(rx-0.04,ry-0.05,0),(rx-0.04,ry-0.06,0),(rx-0.04,ry-0.07,0),
    ]
    face = []
    for i in range(100):
        a = i/100*2*math.pi
        face.append((0.50+0.07*math.cos(a), 0.15+0.06*math.sin(a), 0))
    for i in range(60):
        a = i/60*2*math.pi
        face.append((0.50+0.04*math.cos(a), 0.15+0.035*math.sin(a), 0))
    return {"pose_landmarks":pose,"left_hand_landmarks":None,
            "right_hand_landmarks":rh,"face_landmarks":face}
