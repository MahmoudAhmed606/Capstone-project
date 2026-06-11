"""
YOLO detection evaluation script.

Runs validation on a YOLO-format dataset and prints per-class
precision, recall, mAP50, mAP50-95 to stdout.

Usage
-----
    python evaluation/evaluate_detection.py \\
        --weights models/proctor_yolo.pt \\
        --data training/data.yaml \\
        --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO detection metrics.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data",    default="training/data.yaml")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--split",   default="val", choices=["val", "test"])
    p.add_argument("--device",  default="cpu")
    p.add_argument("--output",  default=None, help="Save JSON metrics to this path")
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.")
        sys.exit(1)

    model = YOLO(args.weights)
    res   = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        device=args.device,
        verbose=True,
    )

    metrics: dict = {}
    try:
        names   = res.names
        metrics["overall"] = {
            "mAP50":     round(float(res.box.map50),  4),
            "mAP50-95":  round(float(res.box.map),    4),
            "precision": round(float(res.box.mp),     4),
            "recall":    round(float(res.box.mr),     4),
        }
        metrics["per_class"] = {}

        header = f"{'Class':<16}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>12}"
        print("\n" + header)
        print("-" * len(header))

        for i, name in names.items():
            p    = float(res.box.p[i])
            r    = float(res.box.r[i])
            ap50 = float(res.box.ap50[i])
            ap   = float(res.box.ap[i])
            print(f"{name:<16}{p:>8.3f}{r:>8.3f}{ap50:>10.3f}{ap:>12.3f}")
            metrics["per_class"][name] = {
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "mAP50":     round(ap50, 4),
                "mAP50-95":  round(ap, 4),
            }

        print("-" * len(header))
        print(
            f"{'Overall':<16}"
            f"{metrics['overall']['precision']:>8.3f}"
            f"{metrics['overall']['recall']:>8.3f}"
            f"{metrics['overall']['mAP50']:>10.3f}"
            f"{metrics['overall']['mAP50-95']:>12.3f}"
        )

    except Exception as exc:
        log.warning("Could not parse per-class metrics: %s", exc)

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))
        log.info("Metrics saved → %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(parse_args())