"""
Smoke tests — fast, no model weights required.

Tests
-----
1. Config files parse without error.
2. YAML thresholds have required keys.
3. EventEngine fires/suppresses events correctly (mock inputs).
4. JSON schema validation on a synthetic session_events payload.
5. Pipeline import and class instantiation (mock models).
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ── Repo root on path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 1. Config loading ──────────────────────────────────────────────────────

class TestConfigLoading:
    def test_default_yaml(self):
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "models"    in cfg
        assert "camera"    in cfg
        assert "inference" in cfg
        assert "session"   in cfg

    def test_thresholds_yaml(self):
        with open("configs/thresholds.yaml") as f:
            th = yaml.safe_load(f)
        assert "events" in th
        events = th["events"]
        for key in ("gaze_away", "multiple_persons", "suspicious_object", "face_missing"):
            assert key in events, f"Missing event key: {key}"

    def test_gaze_away_thresholds(self):
        with open("configs/thresholds.yaml") as f:
            th = yaml.safe_load(f)
        ga = th["events"]["gaze_away"]
        assert ga["duration_seconds"] > 0
        assert ga["yaw_threshold_deg"] > 0


# ── 2. EventEngine unit tests ──────────────────────────────────────────────

class TestEventEngine:
    def _make_engine(self):
        from inference.event_engine import EventEngine
        return EventEngine()   # default thresholds

    def _obj_mock(self, num_persons=1, suspicious=None):
        """Minimal ObjectResult-like mock."""
        m = MagicMock()
        m.num_persons = num_persons
        m.suspicious_objects = suspicious or []
        m.detections = []
        return m

    def test_face_missing_fires(self):
        engine     = self._make_engine()
        face_state = {"num_faces": 0, "brightness": 200.0, "sharpness": 300.0}

        with patch("inference.event_engine.time") as mt:
            # First call: onset recorded at t=0
            mt.monotonic.return_value = 0.0
            events = engine.detect_events(None, None, face_state, 0)
            assert "face_missing" not in [e.type for e in events], \
                "Should not fire before duration elapsed"

            # Second call: t=2.0 exceeds min_duration=1.5 s — event must fire
            mt.monotonic.return_value = 2.0
            events = engine.detect_events(None, None, face_state, 1)

        assert "face_missing" in [e.type for e in events]

    def test_multiple_persons_fires_immediately(self):
        engine = self._make_engine()
        det = MagicMock()
        det.is_person = True
        det.confidence = 0.85
        obj = self._obj_mock(num_persons=2)
        obj.detections = [det, det]
        face_state = {"num_faces": 2, "brightness": 200.0, "sharpness": 300.0}
        events = engine.detect_events(None, obj, face_state, 0)
        types_ = [e.type for e in events]
        assert "multiple_persons" in types_

    def test_gaze_within_threshold_no_event(self):
        engine = self._make_engine()
        face_state = {"num_faces": 1, "brightness": 200.0, "sharpness": 300.0}
        gaze = (5.0, 3.0)   # well within 30° / 25°
        events = engine.detect_events(gaze, None, face_state, 0)
        types_ = [e.type for e in events]
        assert "gaze_away" not in types_

    def test_cooldown_suppresses_repeat(self):
        engine = self._make_engine()
        det_susp = MagicMock()
        det_susp.is_suspicious = True
        det_susp.class_name = "mobile"
        det_susp.confidence = 0.90
        det_susp.bbox = (10, 10, 50, 50)
        obj = self._obj_mock(suspicious=[det_susp])
        obj.detections = []
        face_state = {"num_faces": 1, "brightness": 200.0, "sharpness": 300.0}

        events1 = engine.detect_events(None, obj, face_state, 0)
        events2 = engine.detect_events(None, obj, face_state, 1)
        assert len([e for e in events2 if e.type == "suspicious_object"]) == 0

    def test_event_to_dict_schema(self):
        from inference.event_engine import EventEngine
        engine = EventEngine()
        det = MagicMock()
        det.is_person = True
        det.confidence = 0.9
        obj = self._obj_mock(num_persons=2)
        obj.detections = [det, det]
        face_state = {"num_faces": 2, "brightness": 200.0, "sharpness": 300.0}
        events = engine.detect_events(None, obj, face_state, 5)
        for e in events:
            d = e.to_dict()
            assert "timestamp"   in d
            assert "type"        in d
            assert "confidence"  in d
            assert "frame_index" in d
            assert "bbox"        in d
            assert "details"     in d


# ── 3. JSON schema validation ──────────────────────────────────────────────

class TestJSONSchema:
    SESSION_SCHEMA = {
        "type": "object",
        "required": ["session_id", "session_start", "events"],
        "properties": {
            "session_id":    {"type": "string"},
            "session_start": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "timestamp", "type", "confidence", "frame_index", "bbox", "details"
                    ],
                    "properties": {
                        "timestamp":   {"type": "string"},
                        "type":        {"type": "string"},
                        "confidence":  {"type": "number"},
                        "frame_index": {"type": "integer"},
                        "bbox":        {"type": ["array", "null"]},
                        "details":     {"type": "object"},
                    },
                },
            },
        },
    }

    def _make_payload(self):
        return {
            "session_id": "abc-123",
            "session_start": "2025-01-01T00:00:00+00:00",
            "events": [
                {
                    "timestamp": "2025-01-01T00:01:00+00:00",
                    "type": "gaze_away",
                    "confidence": 0.82,
                    "frame_index": 42,
                    "bbox": None,
                    "details": {"yaw_deg": 35.0, "pitch_deg": 5.0},
                }
            ],
        }

    def test_valid_payload(self):
        import jsonschema
        payload = self._make_payload()
        jsonschema.validate(payload, self.SESSION_SCHEMA)   # raises if invalid

    def test_missing_session_id_fails(self):
        import jsonschema
        payload = self._make_payload()
        del payload["session_id"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, self.SESSION_SCHEMA)

    def test_event_missing_confidence_fails(self):
        import jsonschema
        payload = self._make_payload()
        del payload["events"][0]["confidence"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, self.SESSION_SCHEMA)


# ── 4. Pipeline startup (mocked models) ───────────────────────────────────

class TestPipelineStartup:
    """
    Instantiate ProctorPipeline with mocked model loaders so the test
    passes without real weights being present.
    """

    def _dummy_config(self):
        return {
            "session":   {"privacy_mode": True, "output_dir": "/tmp/pv_test"},
            "camera":    {"source": 0, "width": 640, "height": 480, "fps": 30},
            "inference": {"frame_skip": 2, "batch_size": 1},
            "models": {
                "mediapipe": {"model_path": "models/fake.task", "num_faces": 2,
                              "min_detection_confidence": 0.5},
                "gaze":      {"model_path": "models/fake.pt",   "device": "cpu", "use_onnx": False},
                "yolo":      {"model_path": "models/fake.pt",   "confidence": 0.45,
                              "iou_threshold": 0.45, "device": "cpu", "use_custom_classes": False},
            },
            "calibration": {"duration_per_point": 1.5, "num_points": 5,
                            "fov_horizontal_deg": 60.0, "fov_vertical_deg": 40.0},
        }

    def _dummy_thresholds(self):
        with open("configs/thresholds.yaml") as f:
            return yaml.safe_load(f)

    @patch("inference.pipeline.MediaPipeFaceAnalyzer")
    @patch("inference.pipeline.GazeAdapter")
    @patch("inference.pipeline.YOLOAdapter")
    def test_pipeline_initialises(self, mock_yolo, mock_gaze, mock_mp):
        from inference.pipeline import ProctorPipeline
        p = ProctorPipeline(self._dummy_config(), self._dummy_thresholds())
        assert p.session_id
        assert p.event_engine is not None

    @patch("inference.pipeline.MediaPipeFaceAnalyzer")
    @patch("inference.pipeline.GazeAdapter")
    @patch("inference.pipeline.YOLOAdapter")
    def test_single_frame_predict_returns_prediction(self, mock_yolo, mock_gaze, mock_mp):
        import numpy as np
        from inference.pipeline import ProctorPipeline

        # Configure mocks
        mp_inst = mock_mp.return_value
        lm_mock = MagicMock()
        lm_mock.num_faces = 0
        lm_mock.faces = []
        mp_inst.extract_landmarks.return_value = lm_mock

        gz_inst = mock_gaze.return_value
        gz_inst.predict_gaze.return_value = None

        yl_inst = mock_yolo.return_value
        obj_mock = MagicMock()
        obj_mock.num_persons = 0
        obj_mock.suspicious_objects = []
        obj_mock.detections = []
        yl_inst.predict_objects.return_value = obj_mock

        pipeline = ProctorPipeline(self._dummy_config(), self._dummy_thresholds())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pred  = pipeline.single_frame_predict(frame, frame_index=0)

        assert pred.frame_index == 0
        assert isinstance(pred.events, list)
        assert isinstance(pred.inference_ms, float)