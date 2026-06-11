"""
MediaPipe face landmark utilities.

Wraps the MediaPipe Face Landmarker task (478-point model: 468 face +
10 iris) to extract per-face landmarks and produce eye-region crops for
downstream gaze estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

log = logging.getLogger(__name__)

# ── Landmark index groups ──────────────────────────────────────────────────
LEFT_EYE_INDICES: List[int] = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_INDICES: List[int] = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_IRIS_INDICES:  List[int] = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES: List[int] = [473, 474, 475, 476, 477]

# Eye corner landmark pairs: (outer_idx, inner_idx)
LEFT_EYE_CORNERS  = (33,  133)
RIGHT_EYE_CORNERS = (362, 263)

# Top/bottom lid landmarks used for vertical iris offset
LEFT_LID_TOP    = 159
LEFT_LID_BOTTOM = 145


@dataclass
class FaceResult:
    """Structured output for one detected face."""
    landmarks_px:   np.ndarray              # (478, 2) pixel coords
    landmarks_norm: np.ndarray              # (478, 2) normalised [0, 1]
    left_eye_crop:  Optional[np.ndarray] = None   # BGR crop
    right_eye_crop: Optional[np.ndarray] = None
    left_eye_bbox:  Optional[Tuple[int, int, int, int]] = None   # x,y,w,h
    right_eye_bbox: Optional[Tuple[int, int, int, int]] = None
    face_bbox:      Optional[Tuple[int, int, int, int]] = None
    visibility_score: float = 1.0


@dataclass
class LandmarkResult:
    """Full output from :meth:`MediaPipeFaceAnalyzer.extract_landmarks`."""
    faces:       List[FaceResult] = field(default_factory=list)
    num_faces:   int = 0
    frame_shape: Tuple[int, int] = (0, 0)   # (H, W)


class MediaPipeFaceAnalyzer:
    """
    Thin, stateless wrapper around MediaPipe Face Landmarker.

    Parameters
    ----------
    model_path:
        Path to ``face_landmarker.task``.
    num_faces:
        Maximum faces to detect per frame.
    min_detection_confidence:
        Detection gate.
    min_tracking_confidence:
        Tracking gate.
    eye_crop_padding:
        Fractional padding added around each eye bounding box.
    """

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        eye_crop_padding: float = 0.35,
    ) -> None:
        self._eye_crop_padding = eye_crop_padding

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Download from:\n"
                "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        log.info("MediaPipe Face Landmarker loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_landmarks(self, frame_bgr: np.ndarray) -> LandmarkResult:
        """
        Run face landmarking on one BGR frame.

        Returns a :class:`LandmarkResult` containing per-face data,
        eye crops, and bounding boxes.
        """
        h, w = frame_bgr.shape[:2]
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        result = LandmarkResult(frame_shape=(h, w))
        if not detection.face_landmarks:
            return result

        for face_lm in detection.face_landmarks:
            result.faces.append(self._parse_face(face_lm, frame_bgr, h, w))

        result.num_faces = len(result.faces)
        return result

    def make_eye_crops(
        self,
        frame_bgr: np.ndarray,
        face_result: FaceResult,
        target_size: int = 64,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(left_crop, right_crop)`` resized to ``target_size``."""
        left  = self._resize_crop(face_result.left_eye_crop,  target_size)
        right = self._resize_crop(face_result.right_eye_crop, target_size)
        return left, right

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_face(self, face_lm, frame_bgr: np.ndarray, h: int, w: int) -> FaceResult:
        lm_norm = np.array([[lm.x, lm.y] for lm in face_lm], dtype=np.float32)
        lm_px   = (lm_norm * np.array([w, h])).astype(np.int32)

        face_bbox  = self._bbox_from_indices(lm_px, list(range(min(len(lm_px), 468))), w, h)
        left_bbox  = self._eye_bbox(lm_px, LEFT_EYE_CORNERS,  w, h)
        right_bbox = self._eye_bbox(lm_px, RIGHT_EYE_CORNERS, w, h)

        # Visibility: fraction of landmarks with small Z depth (facing camera)
        vis = float(np.mean([lm.z < 0.05 for lm in face_lm]))

        return FaceResult(
            landmarks_px=lm_px,
            landmarks_norm=lm_norm,
            left_eye_crop=self._crop(frame_bgr, left_bbox),
            right_eye_crop=self._crop(frame_bgr, right_bbox),
            left_eye_bbox=left_bbox,
            right_eye_bbox=right_bbox,
            face_bbox=face_bbox,
            visibility_score=vis,
        )

    def _eye_bbox(
        self,
        lm_px: np.ndarray,
        corner_indices: Tuple[int, int],
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        a_idx, b_idx = corner_indices
        if max(a_idx, b_idx) >= len(lm_px):
            return (0, 0, 0, 0)
        a, b   = lm_px[a_idx], lm_px[b_idx]
        eye_w  = max(abs(int(b[0]) - int(a[0])), 20)
        cx, cy = (int(a[0]) + int(b[0])) // 2, (int(a[1]) + int(b[1])) // 2
        pad    = int(eye_w * self._eye_crop_padding)
        x1 = max(0, cx - eye_w - pad)
        y1 = max(0, cy - eye_w - pad)
        x2 = min(w, cx + eye_w + pad)
        y2 = min(h, cy + eye_w + pad)
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _bbox_from_indices(
        lm_px: np.ndarray, indices: List[int], w: int, h: int
    ) -> Tuple[int, int, int, int]:
        pts = lm_px[indices]
        x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return None
        return frame[y: y + bh, x: x + bw].copy()

    @staticmethod
    def _resize_crop(crop: Optional[np.ndarray], size: int) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)