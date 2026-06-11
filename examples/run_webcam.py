"""
Quickstart: run ProctorVision on a live webcam.

    python examples/run_webcam.py

Press 'q' or ESC to stop.
"""
# Suppress MediaPipe telemetry noise — must be before importing mediapipe
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from inference.threaded_runner import threaded_webcam_runner


def on_event(events):
    for e in events:
        print(f"  EVENT  {e.type:<22}  conf={e.confidence:.2f}  frame={e.frame_index}")


if __name__ == "__main__":
    summary = threaded_webcam_runner(
        config_path="configs/default.yaml",
        thresholds_path="configs/thresholds.yaml",
        show_ui=True,
        event_callback=on_event,
    )
    print("\n=== Session summary ===")
    import json
    print(json.dumps(summary, indent=2))