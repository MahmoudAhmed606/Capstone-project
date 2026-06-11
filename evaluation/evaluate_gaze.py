"""
Gaze estimation evaluation.

Runs L2CS-Net on a directory of face crop images with ground-truth
yaw/pitch labels (CSV) and reports MAE per axis.

CSV format (no header):
    image_path,yaw_deg,pitch_deg

Usage
-----
    python evaluation/evaluate_gaze.py \\
        --model models/l2cs_net.pt \\
        --labels datasets/gaze_eval.csv \\
        --output outputs/gaze_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate L2CS-Net gaze estimation.")
    p.add_argument("--model",  required=True, help="Path to l2cs_net.pt or .onnx")
    p.add_argument("--labels", required=True, help="CSV: image_path,yaw,pitch")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None, help="Save JSON results")
    p.add_argument("--onnx",   action="store_true")
    return p.parse_args()


def _load_labels(csv_path: str) -> List[Tuple[str, float, float]]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                rows.append((row[0].strip(), float(row[1]), float(row[2])))
    return rows


def evaluate(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference.gaze_adapter import GazeAdapter

    adapter = GazeAdapter(
        model_path=args.model,
        device=args.device,
        use_onnx=args.onnx,
    )
    labels = _load_labels(args.labels)
    log.info("Evaluating %d samples …", len(labels))

    yaw_errs:   List[float] = []
    pitch_errs: List[float] = []
    failures = 0

    for img_path, gt_yaw, gt_pitch in labels:
        frame = cv2.imread(img_path)
        if frame is None:
            log.warning("Cannot read %s — skipping", img_path)
            failures += 1
            continue

        pred = adapter.predict_gaze(frame)
        if pred is None:
            failures += 1
            continue

        pr_yaw, pr_pitch = pred
        yaw_errs.append(abs(pr_yaw   - gt_yaw))
        pitch_errs.append(abs(pr_pitch - gt_pitch))

    if not yaw_errs:
        log.error("No valid predictions.")
        return

    mae_yaw   = float(np.mean(yaw_errs))
    mae_pitch = float(np.mean(pitch_errs))
    mae_avg   = (mae_yaw + mae_pitch) / 2.0

    results = {
        "num_samples":   len(labels),
        "num_evaluated": len(yaw_errs),
        "num_failures":  failures,
        "MAE_yaw_deg":   round(mae_yaw,   4),
        "MAE_pitch_deg": round(mae_pitch, 4),
        "MAE_avg_deg":   round(mae_avg,   4),
    }

    print("\n=== Gaze Evaluation ===")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log.info("Results saved → %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(parse_args())