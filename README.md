# ProctorVision

Local, privacy-first AI proctoring — no cloud, no data leaving the machine.

## Model stack

| Model | Purpose | Backend |
|---|---|---|
| MediaPipe Face Landmarker | 478-point face landmarks + iris | MediaPipe Tasks |
| L2CS-Net (ResNet-50) | Gaze estimation (yaw + pitch) | PyTorch or ONNX |
| YOLO11n / YOLO26n | Person + object detection | Ultralytics |

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-org/proctor_vision
cd proctor_vision
pip install -r requirements.txt

# 2. Download model weights
#    MediaPipe face landmarker:
wget -P models/ \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

#    L2CS-Net (ResNet-50, Gaze360):
#    https://github.com/Ahmednull/L2CS-Net → releases → l2cs_net.pt → models/

#    YOLO11n auto-downloads on first run, or place weights manually:
#    cp path/to/yolo11n.pt models/

# 3. Run live webcam
python examples/run_webcam.py

# 4. Run on a video file + generate report
python examples/run_video.py --video exam.mp4 --pdf
```

## API server

```bash
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

## Docker

```bash
# Build
docker build -t proctor-vision .

# Run API server
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  proctor-vision

# Run on a video file
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/videos:/data \
  proctor-vision \
  python examples/run_video.py --video /data/exam.mp4
```

## Fine-tune YOLO on your dataset

```bash
# Prepare dataset in YOLO format under datasets/proctor/
python training/train_yolo.py \
    --data training/data.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --device cpu

# Export to ONNX
python training/export_yolo.py \
    --weights runs/train/proctor_yolo/weights/best.pt

# Update config
# configs/default.yaml → models.yolo.model_path: models/proctor_yolo.pt
#                        models.yolo.use_custom_classes: true
```

## Calibration

At session start, the user is shown five reference dots (centre + four
corners). Raw gaze predictions are averaged per dot and compared to
expected screen-to-angle values. The resulting yaw/pitch bias is applied
to all subsequent predictions for the session.

## Privacy

`configs/default.yaml` → `session.privacy_mode`:

- `true`  — frames processed in memory only, no images written.
- `false` — low-resolution (≤ 320 px) violation snapshots saved to
            `outputs/` when confidence ≥ `snapshot.min_confidence_for_save`.

## Detected events

| Event | Trigger |
|---|---|
| `gaze_away` | Yaw > 30° or pitch > 25° for > 2 s |
| `multiple_persons` | More than 1 person detected |
| `suspicious_object` | Phone, paper, calculator, book, earphones, laptop |
| `face_missing` | No face detected for > 1.5 s |
| `low_visibility` | Brightness < 40 or Laplacian variance < 80 |
| `object_detected` | Any non-person, non-suspicious object |

All thresholds are configurable in `configs/thresholds.yaml`.

## Tests

```bash
pytest tests/ -v
```

## Repo structure