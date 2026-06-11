# ProctorVision — Deliverables Checklist

## Core inference

- [x] `inference/pipeline.py`          — `single_frame_predict()`, `session_runner()`, `save_session_log()`
- [x] `inference/threaded_runner.py`   — `threaded_webcam_runner()`, three-thread producer-consumer
- [x] `inference/gaze_adapter.py`      — L2CS-Net PyTorch + ONNX + landmark heuristic fallback
- [x] `inference/yolo_adapter.py`      — Ultralytics YOLO (model-agnostic)
- [x] `inference/event_engine.py`      — Duration filter, cooldown, confidence gating
- [x] `inference/api_server.py`        — FastAPI REST endpoints

## Utilities

- [x] `utils/mediapipe_utils.py`       — `extract_landmarks()`, `make_eye_crops()`
- [x] `utils/calibration.py`           — 5-point interactive + headless calibration
- [x] `utils/visualization.py`         — `draw_landmarks()`, `draw_gaze_vector()`, event panel

## Training

- [x] `training/data.yaml`             — 7-class dataset config
- [x] `training/train_yolo.py`         — Fine-tune with per-class validation metrics
- [x] `training/export_yolo.py`        — ONNX + OpenVINO export

## Evaluation

- [x] `evaluation/evaluate_detection.py` — P / R / mAP50 / mAP50-95 per class
- [x] `evaluation/evaluate_gaze.py`      — MAE yaw + pitch

## Reports

- [x] `reports/generate_report.py`     — HTML + optional PDF (WeasyPrint)
- [x] `reports/templates/report.html.j2` — Jinja2 template with event log + bar chart

## Testing

- [x] `tests/test_smoke.py`            — Config, EventEngine, JSON schema, pipeline startup

## Deployment

- [x] `Dockerfile`                     — CPU-only, API server default CMD
- [x] `requirements.txt`               — All Python dependencies
- [x] `README.md`                      — Setup, run, fine-tune, Docker, OpenVINO
- [x] `MODEL_CARD.md`                  — Intended use, bias, privacy
- [x] `DELIVERABLES.md`               — This file

## Detected events

| Event | Status |
|---|---|
| `gaze_away` | duration filter + yaw/pitch threshold |
| `multiple_persons` | immediate, cooldown |
| `suspicious_object` | configurable class list, cooldown |
| `object_detected` | generic, lower priority |
| `face_missing` | duration filter |
| `low_visibility` | brightness + Laplacian |

## Optional acceleration

| Backend | Status |
|---|---|
| ONNX Runtime | Supported — set `use_onnx: true` in config |
| OpenVINO | Supported — `export_yolo.py --openvino`, `use_openvino: true` |
| TensorRT | Noted — requires NVIDIA GPU; `model.export(format="engine")` |

## JSON output schema

```json
{
  "session_id": "string",
  "session_start": "ISO-8601",
  "events": [
    {
      "timestamp":   "ISO-8601",
      "type":        "gaze_away | multiple_persons | ...",
      "confidence":  0.0,
      "frame_index": 0,
      "bbox":        [x, y, w, h],
      "details":     {}
    }
  ]
}
```