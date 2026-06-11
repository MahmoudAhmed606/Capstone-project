"""
Producer-consumer threaded webcam runner — performance-tuned.

Performance changes vs original
--------------------------------
- Thread A: sets CAP_PROP_BUFFERSIZE=1 so OpenCV never queues stale frames.
- Thread B: removes the redundant frame.copy() before inference — the
  pipeline already works on its own downscaled copy internally.
- Thread C: snapshot copy only happens on actual violation, not every frame.
- _render_ui: draws on the stored display frame, skips if no new frame.
- FPS measured over a rolling 1-second window and shown in the UI panel.

Thread layout
-------------
Thread A  CaptureThread  — grabs frames at full webcam FPS → frame_queue
Thread B  InferThread    — runs pipeline.single_frame_predict → result_queue
Thread C  EventThread    — fires events, saves snapshots, runs callback
Main thread              — renders OpenCV window (never touches inference)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .pipeline import ProctorPipeline, FramePrediction, load_pipeline
from utils.visualization import draw_event_panel, draw_gaze_vector, draw_bbox

log = logging.getLogger(__name__)

_STOP = object()


class ThreadedWebcamRunner:
    """
    Real-time threaded proctoring runner.

    Parameters
    ----------
    pipeline : ProctorPipeline
    camera_source : int | str
    frame_queue_size : int
    result_queue_size : int
    event_callback : callable | None
        Called from Thread C with a list of ProctoringEvent objects.
    show_ui : bool
    """

    def __init__(
        self,
        pipeline:          ProctorPipeline,
        camera_source:     Any = 0,
        frame_queue_size:  int = 2,    # smaller = less latency
        result_queue_size: int = 4,
        event_callback:    Optional[Callable] = None,
        show_ui:           bool = True,
    ) -> None:
        self.pipeline       = pipeline
        self.camera_source  = camera_source
        self.event_callback = event_callback
        self.show_ui        = show_ui

        self._fq: queue.Queue = queue.Queue(maxsize=frame_queue_size)
        self._rq: queue.Queue = queue.Queue(maxsize=result_queue_size)

        self._stop    = threading.Event()
        self._threads: List[threading.Thread] = []

        self._latest_pred:    Optional[FramePrediction] = None
        self._latest_frame:   Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._capture_fps   = 0.0
        self._inference_fps = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        config_path:     str = "configs/default.yaml",
        thresholds_path: str = "configs/thresholds.yaml",
        **kwargs,
    ) -> "ThreadedWebcamRunner":
        pipeline = load_pipeline(config_path, thresholds_path)
        cam      = pipeline.config.get("camera", {}).get("source", 0)
        return cls(pipeline=pipeline, camera_source=cam, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._thread_a_capture,
                             name="CaptureThread", daemon=True),
            threading.Thread(target=self._thread_b_inference,
                             name="InferThread",   daemon=True),
            threading.Thread(target=self._thread_c_events,
                             name="EventThread",   daemon=True),
        ]
        for t in self._threads:
            t.start()
        log.info("ThreadedWebcamRunner started (%d threads).", len(self._threads))

    def stop(self) -> None:
        log.info("Stopping runner …")
        self._stop.set()
        for q in (self._fq, self._rq):
            try:    q.put_nowait(_STOP)
            except queue.Full: pass
        for t in self._threads:
            t.join(timeout=6.0)
        self.pipeline.close()
        if self.show_ui:
            cv2.destroyAllWindows()
        log.info("Runner stopped.")

    def run_blocking(self) -> Dict[str, Any]:
        """Start and block until 'q' / ESC. Returns session summary."""
        self.start()
        try:
            while not self._stop.is_set():
                if self.show_ui:
                    self._render_ui()
                    key = cv2.waitKey(1) & 0xFF   # 1 ms — keeps UI snappy
                    if key in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return self.pipeline.save_session_log()

    # ------------------------------------------------------------------
    # Thread A — Webcam capture (runs at full camera FPS)
    # ------------------------------------------------------------------

    def _thread_a_capture(self) -> None:
        cam = self.pipeline.config.get("camera", {})
        cap = cv2.VideoCapture(self.camera_source)

        # Critical: buffersize=1 ensures we always get the LATEST frame,
        # never a stale one that's been sitting in OpenCV's internal queue.
        cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam.get("width",  640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
        cap.set(cv2.CAP_PROP_FPS,          cam.get("fps",     30))

        if not cap.isOpened():
            log.error("Thread A: cannot open camera %s", self.camera_source)
            self._stop.set()
            return

        log.info("Thread A: capturing from source %s", self.camera_source)
        n, t0 = 0, time.monotonic()

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            n += 1
            dt = time.monotonic() - t0
            if dt >= 1.0:
                self._capture_fps = n / dt
                n, t0 = 0, time.monotonic()

            # Non-blocking put: drop oldest frame if queue is full
            # This ensures Thread B always works on the most recent frame.
            try:
                self._fq.put_nowait(frame)
            except queue.Full:
                try:
                    self._fq.get_nowait()    # discard oldest
                    self._fq.put_nowait(frame)
                except queue.Empty:
                    pass

        cap.release()
        log.info("Thread A: stopped.")

    # ------------------------------------------------------------------
    # Thread B — Inference (MediaPipe → gaze → YOLO → events)
    # ------------------------------------------------------------------

    def _thread_b_inference(self) -> None:
        log.info("Thread B: inference loop started.")
        skip      = self.pipeline.config.get("inference", {}).get("frame_skip", 1)
        raw, n, t0 = 0, 0, time.monotonic()

        while not self._stop.is_set():
            try:
                item = self._fq.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _STOP:
                break

            raw += 1
            if skip > 1 and raw % skip != 0:
                continue

            pred = self.pipeline.single_frame_predict(item, raw)

            n  += 1
            dt  = time.monotonic() - t0
            if dt >= 1.0:
                self._inference_fps = n / dt
                n, t0 = 0, time.monotonic()

            # Store latest frame for UI — no copy unless snapshot needed
            with self._lock:
                self._latest_pred  = pred
                self._latest_frame = item   # no .copy() — UI reads only

            try:
                self._rq.put_nowait(pred)
            except queue.Full:
                try:
                    self._rq.get_nowait()
                    self._rq.put_nowait(pred)
                except queue.Empty:
                    pass

        log.info("Thread B: stopped.")

    # ------------------------------------------------------------------
    # Thread C — Event processing + snapshots
    # ------------------------------------------------------------------

    def _thread_c_events(self) -> None:
        log.info("Thread C: event loop started.")
        privacy   = self.pipeline.config.get("session", {}).get("privacy_mode", True)
        snap_cfg  = self.pipeline.thresholds.get("snapshot", {})
        min_conf  = snap_cfg.get("min_confidence_for_save", 0.75)
        max_res   = snap_cfg.get("max_resolution", 320)
        save_snap = snap_cfg.get("save_on_violation", True)

        while not self._stop.is_set():
            try:
                item = self._rq.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _STOP:
                break

            pred: FramePrediction = item

            if pred.events:
                log.debug("Frame %d: %s",
                          pred.frame_index, [e.type for e in pred.events])
                if self.event_callback:
                    try:
                        self.event_callback(pred.events)
                    except Exception as exc:
                        log.warning("event_callback error: %s", exc)

                # Snapshot: only copy frame when actually saving
                if not privacy and save_snap:
                    high = [e for e in pred.events if e.confidence >= min_conf]
                    if high:
                        with self._lock:
                            snap = (self._latest_frame.copy()
                                    if self._latest_frame is not None else None)
                        if snap is not None:
                            self._save_snapshot(snap, high[0], max_res)

        log.info("Thread C: stopped.")

    # ------------------------------------------------------------------
    # UI — runs on main thread only
    # ------------------------------------------------------------------

    def _render_ui(self) -> None:
        with self._lock:
            pred  = self._latest_pred
            frame = self._latest_frame

        if frame is None:
            return

        display = frame.copy()   # copy only for drawing — never blocks inference

        if pred:
            fps = self._inference_fps
            draw_event_panel(display, [e.type for e in pred.events], fps)

            if pred.landmark_result and pred.landmark_result.num_faces > 0:
                face0 = pred.landmark_result.faces[0]
                if pred.gaze_result and face0.face_bbox:
                    x, y, bw, bh = face0.face_bbox
                    draw_gaze_vector(
                        display,
                        (x + bw // 2, y + bh // 2),
                        *pred.gaze_result,
                    )

            if pred.object_result:
                for det in pred.object_result.detections:
                    draw_bbox(
                        display, det.bbox,
                        f"{det.class_name} {det.confidence:.0%}",
                    )

        cv2.imshow("ProctorVision — Live", display)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(
        self, frame: np.ndarray, event: Any, max_res: int
    ) -> None:
        h, w  = frame.shape[:2]
        scale = min(max_res / w, max_res / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        out = (self.pipeline.output_dir
               / f"snap_{event.type}_{event.frame_index}.jpg")
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        log.info("Snapshot → %s", out)


# ── One-liner entry point ──────────────────────────────────────────────────

def threaded_webcam_runner(
    config_path:     str = "configs/default.yaml",
    thresholds_path: str = "configs/thresholds.yaml",
    show_ui:         bool = True,
    event_callback:  Optional[Callable] = None,
) -> Dict[str, Any]:
    """Start threaded runner, block until 'q' / ESC, return session summary."""
    runner = ThreadedWebcamRunner.from_configs(
        config_path=config_path,
        thresholds_path=thresholds_path,
        show_ui=show_ui,
        event_callback=event_callback,
    )
    return runner.run_blocking()