"""
Fine-tune YOLO on the proctoring dataset (data.yaml).

Usage
-----
    python training/train_yolo.py \\
        --data training/data.yaml \\
        --model yolo11n.pt \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16 \\
        --project runs/train \\
        --name proctor_yolo

After training the best weights are at:
    runs/train/proctor_yolo/weights/best.pt

Copy them to models/ and update configs/default.yaml:
    models.yolo.model_path: models/proctor_yolo.pt
    models.yolo.use_custom_classes: true
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for proctoring detection.")
    p.add_argument("--data",    default="training/data.yaml", help="Dataset YAML")
    p.add_argument("--model",   default="yolo11n.pt",          help="Base weights (pt or yaml)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device",  default="cpu", help="'cpu', '0', '0,1' …")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name",    default="proctor_yolo")
    p.add_argument("--resume",  action="store_true", help="Resume from last checkpoint")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not Path(args.data).exists():
        log.error("Dataset YAML not found: %s", args.data)
        sys.exit(1)

    log.info("Loading base model: %s", args.model)
    model = YOLO(args.model)

    log.info(
        "Training — epochs=%d  imgsz=%d  batch=%d  device=%s",
        args.epochs, args.imgsz, args.batch, args.device,
    )

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        save=True,
        plots=True,
        # Augmentation from data.yaml hints
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    log.info("Training complete. Best weights → %s", best)

    # Print per-class metrics
    if hasattr(results, "results_dict"):
        log.info("Metrics: %s", results.results_dict)

    print("\n=== Validation metrics on best weights ===")
    model_best = YOLO(str(best))
    val_res    = model_best.val(data=args.data, imgsz=args.imgsz, device=args.device)
    _print_metrics(val_res)


def _print_metrics(val_res) -> None:
    """Print precision, recall, mAP50, mAP50-95 per class."""
    try:
        names = val_res.names
        for i, name in names.items():
            p     = val_res.box.p[i]  if hasattr(val_res.box, "p")  else "—"
            r     = val_res.box.r[i]  if hasattr(val_res.box, "r")  else "—"
            ap50  = val_res.box.ap50[i] if hasattr(val_res.box, "ap50") else "—"
            ap    = val_res.box.ap[i]   if hasattr(val_res.box, "ap")   else "—"
            print(
                f"  {name:<14}  P={p:.3f}  R={r:.3f}  "
                f"mAP50={ap50:.3f}  mAP50-95={ap:.3f}"
            )
        print(
            f"\n  Overall  mAP50={val_res.box.map50:.4f}"
            f"  mAP50-95={val_res.box.map:.4f}"
        )
    except Exception as exc:
        log.warning("Could not parse detailed metrics: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(parse_args())