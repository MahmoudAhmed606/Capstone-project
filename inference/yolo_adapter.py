"""
YOLO object detection adapter.

Wraps Ultralytics YOLO for person counting and suspicious-object
detection.  Default model is YOLO11n; code is designed so YOLO26n
(or any Ultralytics-compatible model) can be swapped in by changing
``model_path`` in ``configs/default.yaml``.

Detected person class:       ``person`` (COCO class 0)
Suspicious object classes:   mobile · paper · calculator · book ·
                             earphones · laptop  (custom fine-tune)
                             cell_phone (COCO class 67, base model)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Custom class names (matches training/data.yaml)
CUSTOM_CLASS_NAMES = [
    "person", "mobile", "paper", "calculator", "book", "earphones", "laptop",
]

# COCO cell phone class id (used when fine-tuned model not available)
COCO_CELL_PHONE_ID = 67

SUSPICIOUS_NAMES = frozenset(
    ["mobile", "cell_phone", "paper", "calculator", "book", "earphones", "laptop"]
)


@dataclass
class Detection:
    """Single object detection."""
    class_id:     int
    class_name:   str
    confidence:   float
    bbox:         tuple        # (x, y, w, h) pixels
    is_person:    bool = False
    is_suspicious: bool = False


@dataclass
class ObjectResult:
    """YOLO inference output for one frame."""
    detections:        List[Detection] = field(default_factory=list)
    num_persons:       int = 0
    suspicious_objects: List[Detection] = field(default_factory=list)
    inference_ms:      float = 0.0


class YOLOAdapter:
    """
    Ultralytics YOLO wrapper.

    Parameters
    ----------
    model_path:
        Path to ``.pt`` or ``.onnx`` weights.
        Auto-downloads ``yolo11n.pt`` if not found.
    confidence:
        Minimum detection confidence.
    iou_threshold:
        NMS IoU threshold.
    input_size:
        YOLO input resolution (square).
    device:
        ``"cpu"`` or ``"cuda"``.
    use_custom_classes:
        ``True`` after fine-tuning on the proctoring dataset.
        ``False`` uses COCO class IDs.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        confidence:    float = 0.45,
        iou_threshold: float = 0.45,
        input_size:    int   = 640,
        device:        str   = "cpu",
        use_custom_classes: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.conf              = confidence
        self.iou               = iou_threshold
        self.imgsz             = input_size
        self.device            = device
        self.use_custom_classes = use_custom_classes

        p = Path(model_path)
        effective_path = str(p) if p.exists() else "yolo11n.pt"
        if not p.exists():
            log.warning(
                "YOLO weights not found at '%s' — Ultralytics will auto-download yolo11n.pt",
                model_path,
            )

        self._model = YOLO(effective_path)
        log.info("YOLO loaded from %s", effective_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_objects(self, frame_bgr: np.ndarray) -> ObjectResult:
        """Run YOLO inference on one BGR frame."""
        t0 = time.monotonic()

        results = self._model(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()
            confs   = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cid in zip(boxes, confs, cls_ids):
                name         = self._class_name(cid)
                is_person    = name == "person" or (
                    not self.use_custom_classes and cid == 0
                )
                is_suspicious = name in SUSPICIOUS_NAMES or (
                    not self.use_custom_classes and cid == COCO_CELL_PHONE_ID
                )
                x1, y1, x2, y2 = box
                detections.append(Detection(
                    class_id=int(cid),
                    class_name=name,
                    confidence=float(conf),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    is_person=is_person,
                    is_suspicious=is_suspicious,
                ))

        persons    = [d for d in detections if d.is_person]
        suspicious = [d for d in detections if d.is_suspicious]

        return ObjectResult(
            detections=detections,
            num_persons=len(persons),
            suspicious_objects=suspicious,
            inference_ms=elapsed_ms,
        )

    def export_onnx(self, output_path: str = "models/yolo11n.onnx") -> None:
        """Export model to ONNX."""
        self._model.export(format="onnx", imgsz=self.imgsz, half=False)
        log.info("YOLO exported to ONNX → %s", output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _class_name(self, class_id: int) -> str:
        if self.use_custom_classes:
            return CUSTOM_CLASS_NAMES[class_id] \
                if 0 <= class_id < len(CUSTOM_CLASS_NAMES) \
                else f"class_{class_id}"
        try:
            return self._model.names[class_id]
        except (KeyError, IndexError):
            return f"class_{class_id}"