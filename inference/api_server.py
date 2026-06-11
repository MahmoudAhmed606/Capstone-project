"""
FastAPI REST API server for ProctorVision.

Endpoints
---------
POST /session/start          — initialise a new session
POST /session/stop           — stop session, return summary
POST /infer                  — single-frame inference (base64 image)
GET  /session/events         — stream all events so far
GET  /session/summary        — current session summary
GET  /health                 — liveness probe

Run
---
    uvicorn inference.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .pipeline import ProctorPipeline, FramePrediction
from .event_engine import ProctoringEvent

log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ProctorVision API",
    description="AI-powered exam proctoring inference service.",
    version="1.0.0",
)

# ── Global state ───────────────────────────────────────────────────────────

_pipeline:     Optional[ProctorPipeline] = None
_pipeline_lock = threading.Lock()
_CONFIG_PATH      = "configs/default.yaml"
_THRESHOLDS_PATH  = "configs/thresholds.yaml"


def _get_pipeline() -> ProctorPipeline:
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=400, detail="No active session. POST /session/start first.")
    return _pipeline


# ── Schemas ────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    config_path:     str = _CONFIG_PATH
    thresholds_path: str = _THRESHOLDS_PATH
    privacy_mode:    Optional[bool] = None


class StartSessionResponse(BaseModel):
    session_id:    str
    session_start: str
    message:       str


class InferRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG frame")
    frame_index: Optional[int] = None


class InferResponse(BaseModel):
    frame_index:  int
    timestamp:    str
    events:       List[Dict[str, Any]]
    num_faces:    int
    gaze_yaw:     Optional[float]
    gaze_pitch:   Optional[float]
    num_persons:  int
    inference_ms: float


class SessionSummaryResponse(BaseModel):
    session_id:    str
    session_start: str
    total_events:  int
    event_counts:  Dict[str, int]


# ── Startup / shutdown ─────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("ProctorVision API server started.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _pipeline
    with _pipeline_lock:
        if _pipeline:
            _pipeline.close()
            _pipeline = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/session/start", response_model=StartSessionResponse)
async def session_start(req: StartSessionRequest) -> StartSessionResponse:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            _pipeline.close()

        with open(req.config_path)     as f: config     = yaml.safe_load(f)
        with open(req.thresholds_path) as f: thresholds = yaml.safe_load(f)

        if req.privacy_mode is not None:
            config.setdefault("session", {})["privacy_mode"] = req.privacy_mode

        _pipeline = ProctorPipeline(config, thresholds)

    return StartSessionResponse(
        session_id=_pipeline.session_id,
        session_start=_pipeline.session_start,
        message="Session started.",
    )


@app.post("/session/stop")
async def session_stop(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            raise HTTPException(status_code=400, detail="No active session.")
        summary = _pipeline.save_session_log()
        background_tasks.add_task(_pipeline.close)
        _pipeline = None
    return summary


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    pipeline = _get_pipeline()

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr       = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    pred: FramePrediction = pipeline.single_frame_predict(frame, req.frame_index)

    gaze_yaw = gaze_pitch = None
    if pred.gaze_result:
        gaze_yaw, gaze_pitch = pred.gaze_result

    return InferResponse(
        frame_index=pred.frame_index,
        timestamp=pred.timestamp,
        events=[e.to_dict() for e in pred.events],
        num_faces=pred.landmark_result.num_faces if pred.landmark_result else 0,
        gaze_yaw=round(gaze_yaw, 2) if gaze_yaw is not None else None,
        gaze_pitch=round(gaze_pitch, 2) if gaze_pitch is not None else None,
        num_persons=pred.object_result.num_persons if pred.object_result else 0,
        inference_ms=round(pred.inference_ms, 2),
    )


@app.get("/session/events")
async def session_events() -> Dict[str, Any]:
    pipeline = _get_pipeline()
    return {
        "session_id": pipeline.session_id,
        "events":     [e.to_dict() for e in pipeline.event_engine.session_events],
        "total":      len(pipeline.event_engine.session_events),
    }


@app.get("/session/summary")
async def session_summary() -> Dict[str, Any]:
    pipeline = _get_pipeline()
    from collections import Counter
    counts = Counter(e.type for e in pipeline.event_engine.session_events)
    return {
        "session_id":    pipeline.session_id,
        "session_start": pipeline.session_start,
        "total_events":  len(pipeline.event_engine.session_events),
        "event_counts":  dict(counts),
    }