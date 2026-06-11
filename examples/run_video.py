"""
Run ProctorVision on a video file and generate a report.

    python examples/run_video.py --video path/to/exam_recording.mp4

Output files land in ./outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from inference.pipeline import load_pipeline
from reports.generate_report import generate_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--thresh", default="configs/thresholds.yaml")
    p.add_argument("--show",   action="store_true", help="Show live preview")
    p.add_argument("--pdf",    action="store_true", help="Generate PDF report")
    args = p.parse_args()

    pipeline = load_pipeline(args.config, args.thresh)
    summary  = pipeline.session_runner(video_source=args.video, show_ui=args.show)

    print("\n=== Session summary ===")
    print(json.dumps(summary, indent=2))

    report = generate_report(
        events_path="outputs/session_events.json",
        summary_path="outputs/session_summary.json",
        output_path="outputs/report.html",
        pdf=args.pdf,
    )
    print(f"\nReport → {report}")


if __name__ == "__main__":
    main()