"""
Demo: send a frame to the running API server.

    # Terminal 1 — start API server
    uvicorn inference.api_server:app --port 8000

    # Terminal 2 — run this demo
    python examples/api_client_demo.py --image path/to/frame.jpg
"""

import argparse
import base64
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    args = p.parse_args()

    # Start session
    r = requests.post(f"{BASE}/session/start", json={})
    r.raise_for_status()
    print("Session:", r.json()["session_id"])

    # Encode image
    img_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()

    # Infer
    r = requests.post(f"{BASE}/infer", json={"image_b64": img_b64})
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

    # Summary
    r = requests.get(f"{BASE}/session/summary")
    print("\nSummary:", json.dumps(r.json(), indent=2))

    # Stop
    r = requests.post(f"{BASE}/session/stop")
    print("\nStopped.")


if __name__ == "__main__":
    main()