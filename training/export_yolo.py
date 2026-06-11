"""
Export fine-tuned YOLO weights to ONNX (and optionally OpenVINO IR).

Usage
-----
    # ONNX only
    python training/export_yolo.py --weights runs/train/proctor_yolo/weights/best.pt

    # ONNX + OpenVINO
    python training/export_yolo.py --weights best.pt --openvino

Output files will be placed in the same directory as --weights.

OpenVINO
--------
Install: pip install openvino
Requires openvino-dev for mo (Model Optimizer) if using older versions.
For openvino>=2024 the YOLO Ultralytics exporter handles this automatically.

TensorRT notes
--------------
TensorRT export requires an NVIDIA GPU and matching TRT install.
Run on the target GPU machine:
    model.export(format="engine", half=True, device=0)
Then load with: YOLO("best.engine")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO to ONNX / OpenVINO.")
    p.add_argument("--weights", required=True,       help="Path to best.pt")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--openvino", action="store_true", help="Also export OpenVINO IR")
    p.add_argument("--dynamic",  action="store_true", help="Dynamic batch axis in ONNX")
    p.add_argument("--half",     action="store_true", help="FP16 (GPU only)")
    return p.parse_args()


def export(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.")
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    model = YOLO(str(weights))

    # ── ONNX ──────────────────────────────────────────────────────────
    log.info("Exporting to ONNX …")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        half=args.half,
        opset=17,
        simplify=True,
    )
    log.info("ONNX → %s", onnx_path)

    # ── OpenVINO IR ───────────────────────────────────────────────────
    if args.openvino:
        log.info("Exporting to OpenVINO IR …")
        try:
            ov_path = model.export(format="openvino", imgsz=args.imgsz, half=False)
            log.info("OpenVINO IR → %s", ov_path)
        except Exception as exc:
            log.error("OpenVINO export failed: %s", exc)
            log.error("Install: pip install openvino>=2024.0")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export(parse_args())