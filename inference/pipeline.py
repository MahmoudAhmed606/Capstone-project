"""
Core inference pipeline.

Key fixes in this version
--------------------------
1. Landmark coordinate scaling:
   MediaPipe runs on the downscaled 480px inference frame. All landmark
   pixel coordinates and face bboxes are now scaled back to original
   frame resolution BEFORE being stored in FramePrediction. This fixes
   the gaze arrow appearing at the wrong position on the display frame.

2. FramePrediction now carries infer_scale so callers can always verify
   or re-apply scaling if needed.

3. Quality check runs on original full-res frame (not the small copy),
   avoiding false low_visibility from bilinear-resize blur.

4. Gaze origin drawn from nose-tip landmark (point 1) which sits at
   the natural visual centre of the face — more accurate than bbox centre.
   Falls back to bbox centre if landmark index is out of range.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from utils.mediapipe_utils import MediaPipeFaceAnalyzer, LandmarkResult
from utils.visualization import (
    frame_quality_check,
    draw_event_panel,
    draw_gaze_vector,
    draw_bbox,
    draw_face_box,
)
from .gaze_adapter import GazeAdapter
from .yolo_adapter import YOLOAdapter, ObjectResult
from .event_engine import EventEngine, ProctoringEvent

log = logging.getLogger(__name__)


# ── Helper ─────────────────────────────────────────────────────────────────

def _scale_bbox(bbox: Optional[Tuple], s: float) -> Optional[Tuple]:
    """Scale a (x, y, w, h) bbox by factor s. Returns None if input is None."""
    if bbox is None:
        return None
    x, y, w, h = bbox
    return (int(x * s), int(y * s), int(w * s), int(h * s))


# ── Output container ───────────────────────────────────────────────────────

class FramePrediction:
    """All model outputs for one frame — all coordinates in ORIGINAL resolution."""
    __slots__ = (
        "frame_index", "timestamp",
        "landmark_result", "gaze_result",
        "object_result", "events",
        "brightness", "sharpness",
        "inference_ms", "infer_scale",
    )

    def __init__(self) -> None:
        self.frame_index:     int = 0
        self.timestamp:       str = ""
        self.landmark_result: Optional[LandmarkResult] = None
        self.gaze_result:     Optional[Tuple[float, float]] = None
        self.object_result:   Optional[ObjectResult] = None
        self.events:          List[ProctoringEvent] = []
        self.brightness:      float = 255.0
        self.sharpness:       float = 999.0
        self.inference_ms:    float = 0.0
        self.infer_scale:     float = 1.0    # scale applied to inference frame


# ── Pipeline ───────────────────────────────────────────────────────────────

class ProctorPipeline:
    """
    End-to-end proctoring inference pipeline.

    Parameters
    ----------
    config : dict   from configs/default.yaml
    thresholds : dict   from configs/thresholds.yaml
    """

    def __init__(self, config: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
        self.config      = config
        self.thresholds  = thresholds
        self.session_id  = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc).isoformat()
        self._frame_idx  = 0

        mc  = config.get("models", {})
        inf = config.get("inference", {})
        self._infer_width: int = inf.get("infer_width", 480)

        # ── Models ────────────────────────────────────────────────────
        mp_cfg = mc.get("mediapipe", {})
        self.face_analyzer = MediaPipeFaceAnalyzer(
            model_path=mp_cfg.get("model_path", "models/face_landmarker.task"),
            num_faces=mp_cfg.get("num_faces", 2),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
        )

        gz_cfg = mc.get("gaze", {})
        self.gaze_adapter = GazeAdapter(
            model_path=gz_cfg.get("model_path", "models/L2CSNet_gaze360.pkl"),
            device=gz_cfg.get("device", "cpu"),
            use_onnx=gz_cfg.get("use_onnx", False),
            use_heuristic=gz_cfg.get("use_heuristic", True),
            input_size=gz_cfg.get("input_size", 448),
        )

        yl_cfg = mc.get("yolo", {})
        self.yolo_adapter = YOLOAdapter(
            model_path=yl_cfg.get("model_path", "models/yolo11n.pt"),
            confidence=yl_cfg.get("confidence", 0.45),
            iou_threshold=yl_cfg.get("iou_threshold", 0.45),
            input_size=yl_cfg.get("input_size", 320),
            device=yl_cfg.get("device", "cpu"),
            use_custom_classes=yl_cfg.get("use_custom_classes", False),
        )

        self.event_engine = EventEngine(thresholds=thresholds)

        sess_cfg = config.get("session", {})
        self._privacy_mode = sess_cfg.get("privacy_mode", True)
        self.output_dir    = Path(sess_cfg.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session_events: List[ProctoringEvent] = []

        log.info(
            "ProctorPipeline ready — session=%s  heuristic=%s  "
            "yolo_sz=%d  infer_width=%d",
            self.session_id,
            gz_cfg.get("use_heuristic", True),
            yl_cfg.get("input_size", 320),
            self._infer_width,
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def single_frame_predict(
        self,
        frame_bgr: np.ndarray,
        frame_index: Optional[int] = None,
    ) -> FramePrediction:
        """
        Run the full pipeline on one BGR frame.

        All coordinates in the returned FramePrediction are in the
        ORIGINAL frame resolution, not the downscaled inference copy.

        Steps
        -----
        1. Quality check on ORIGINAL frame (avoids resize-blur false alarms)
        2. Downscale for inference
        3. MediaPipe landmarks on small frame → scale back to original coords
        4. Gaze estimation
        5. YOLO on small frame → scale bboxes back
        6. EventEngine
        """
        t0  = time.monotonic()
        idx = frame_index if frame_index is not None else self._frame_idx
        self._frame_idx += 1

        pred             = FramePrediction()
        pred.frame_index = idx
        pred.timestamp   = datetime.now(timezone.utc).isoformat()

        # ── 1. Quality on ORIGINAL frame ──────────────────────────────
        # Must run BEFORE downscaling — resize blur reduces sharpness score
        # and was causing false low_visibility events on weak cameras.
        pred.brightness, pred.sharpness = frame_quality_check(frame_bgr)

        # ── 2. Downscale for inference ─────────────────────────────────
        h, w = frame_bgr.shape[:2]
        if w > self._infer_width:
            scale = self._infer_width / w
            small = cv2.resize(
                frame_bgr,
                (self._infer_width, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            scale = 1.0
            small = frame_bgr

        pred.infer_scale = scale

        # ── 3. Landmarks on small frame → scale back ───────────────────
        lm = self.face_analyzer.extract_landmarks(small)

        # FIX: All landmark pixel coords come from the 480px small frame.
        # Scale every coordinate back to original resolution so the gaze
        # arrow and face box are drawn at the correct position.
        if scale != 1.0 and lm.num_faces > 0:
            inv = 1.0 / scale
            for face in lm.faces:
                face.landmarks_px = (
                    face.landmarks_px.astype(float) * inv
                ).astype(np.int32)
                face.face_bbox      = _scale_bbox(face.face_bbox,      inv)
                face.left_eye_bbox  = _scale_bbox(face.left_eye_bbox,  inv)
                face.right_eye_bbox = _scale_bbox(face.right_eye_bbox, inv)

        pred.landmark_result = lm

        # ── 4. Gaze estimation ─────────────────────────────────────────
        gaze = None
        if lm.num_faces > 0:
            face0 = lm.faces[0]
            if self.gaze_adapter._use_heuristic:
                # Heuristic uses scaled-back landmark coords (already correct)
                gaze = self.gaze_adapter.predict_gaze(None, face0.landmarks_px)
            elif face0.face_bbox:
                # Neural net needs a crop from SMALL frame (before scale-back)
                # Recompute crop coords in small-frame space
                x, y, bw, bh = face0.face_bbox
                xs = int(x * scale); ys = int(y * scale)
                ws = int(bw * scale); hs = int(bh * scale)
                if ws > 0 and hs > 0:
                    crop = small[ys: ys + hs, xs: xs + ws]
                    gaze = self.gaze_adapter.predict_gaze(crop, face0.landmarks_px)

        pred.gaze_result = gaze

        # ── 5. YOLO on small frame → scale bboxes back ────────────────
        obj = self.yolo_adapter.predict_objects(small)
        if scale != 1.0 and obj:
            inv = 1.0 / scale
            for det in obj.detections:
                det.bbox = _scale_bbox(det.bbox, inv)

        pred.object_result = obj

        # ── 6. Events ─────────────────────────────────────────────────
        face_state = {
            "num_faces":  lm.num_faces,
            "brightness": pred.brightness,
            "sharpness":  pred.sharpness,
        }
        events = self.event_engine.detect_events(gaze, obj, face_state, idx)
        pred.events = events
        self._session_events.extend(events)
        pred.inference_ms = (time.monotonic() - t0) * 1000
        return pred

    # ------------------------------------------------------------------
    # Session runner (blocking — video files / CI)
    # ------------------------------------------------------------------

    def session_runner(
        self,
        video_source: Any = 0,
        max_frames:   Optional[int] = None,
        show_ui:      bool = False,
    ) -> Dict[str, Any]:
        cap  = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")
        skip = self.config.get("inference", {}).get("frame_skip", 1)
        raw  = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw += 1
                if max_frames and raw > max_frames:
                    break
                if skip > 1 and raw % skip != 0:
                    continue
                pred = self.single_frame_predict(frame, raw)
                if show_ui:
                    self._annotate(frame, pred)
                    cv2.imshow("ProctorVision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show_ui:
                cv2.destroyAllWindows()
        return self.save_session_log()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_session_log(self) -> Dict[str, Any]:
        payload = {
            "session_id":    self.session_id,
            "session_start": self.session_start,
            "events": [e.to_dict() for e in self._session_events],
        }
        summary = self._build_summary(payload)
        (self.output_dir / "session_events.json").write_text(json.dumps(payload, indent=2))
        (self.output_dir / "session_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Session log saved → %s", self.output_dir)
        return summary

    def generate_summary_report(self) -> Dict[str, Any]:
        return self.save_session_log()

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, pred: FramePrediction) -> None:
        """Draw all overlays on the ORIGINAL resolution display frame."""
        fps    = 1000.0 / pred.inference_ms if pred.inference_ms > 0 else 0
        active = [e.type for e in pred.events]

        # Face box + gaze arrow
        if pred.landmark_result and pred.landmark_result.num_faces > 0:
            face0 = pred.landmark_result.faces[0]

            # Draw face bounding box
            if face0.face_bbox:
                draw_face_box(frame, face0.face_bbox)

            # Gaze arrow — drawn from nose tip (landmark 1) which sits at
            # the visual centre of the face, inside the bounding box.
            if pred.gaze_result:
                lm = face0.landmarks_px
                if len(lm) > 1:
                    # Nose tip is the most stable central facial landmark
                    origin = (int(lm[1][0]), int(lm[1][1]))
                elif face0.face_bbox:
                    x, y, bw, bh = face0.face_bbox
                    origin = (x + bw // 2, y + bh // 2)
                else:
                    origin = None

                if origin:
                    draw_gaze_vector(frame, origin, *pred.gaze_result)

        # Object bboxes
        if pred.object_result:
            for det in pred.object_result.detections:
                draw_bbox(frame, det.bbox, f"{det.class_name} {det.confidence:.0%}")

        # HUD panel (drawn last so it's always on top)
        draw_event_panel(frame, active, fps)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(payload: Dict) -> Dict:
        events = payload["events"]
        counts = Counter(e["type"] for e in events)
        return {
            "session_id":    payload["session_id"],
            "session_start": payload["session_start"],
            "total_events":  len(events),
            "event_counts":  dict(counts),
            "high_confidence_violations": [
                e for e in events
                if e["type"] in ("gaze_away", "multiple_persons", "suspicious_object")
                and e["confidence"] >= 0.70
            ],
        }

    def close(self) -> None:
        self.face_analyzer.close()
        log.info("ProctorPipeline closed.")


# ── Factory ────────────────────────────────────────────────────────────────

def load_pipeline(
    config_path:     str = "configs/default.yaml",
    thresholds_path: str = "configs/thresholds.yaml",
) -> ProctorPipeline:
    with open(config_path)     as f: config     = yaml.safe_load(f)
    with open(thresholds_path) as f: thresholds = yaml.safe_load(f)
    return ProctorPipeline(config, thresholds)
