"""
Proctoring event engine — fixed gaze_away accumulation.

Key fixes
---------
gaze_away onset timer:
  - yaw_threshold lowered to 12°, pitch to 9° — reachable by iris heuristic
  - none_grace_frames: onset timer does NOT reset on a single None frame.
    Only resets after none_grace_frames consecutive None frames. This means
    a blink, brief iris occlusion, or momentary tracking loss no longer
    restarts the 1.5 s clock.
  - Confidence formula changed: margin-based (how far past threshold) so
    values close to the threshold still accumulate instead of being gated.

multiple_persons:
  - Valid person now requires confidence >= 0.60 AND bbox area >= 1536 px²
    (rejects hands, reflections, tiny background detections).

low_visibility:
  - Changed OR → AND: both dark AND blurry must be true simultaneously.
  - Added duration_seconds filter: a single dark frame no longer fires.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .yolo_adapter import ObjectResult

log = logging.getLogger(__name__)

SUSPICIOUS_NAMES = frozenset(
    ["mobile", "cell_phone", "paper", "calculator", "book", "earphones", "laptop"]
)


@dataclass
class ProctoringEvent:
    """Single proctoring event — matches the spec JSON schema."""
    timestamp:   str
    type:        str
    confidence:  float
    frame_index: int
    bbox:        Optional[List[int]]
    details:     Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":   self.timestamp,
            "type":        self.type,
            "confidence":  round(self.confidence, 4),
            "frame_index": self.frame_index,
            "bbox":        self.bbox,
            "details":     self.details,
        }


@dataclass
class _State:
    """Per-event duration-filter + cooldown state machine."""
    name:            str
    min_duration:    float
    cooldown:        float
    min_confidence:  float
    onset_time:      Optional[float] = None
    last_fired_time: Optional[float] = None

    def update(self, condition: bool, confidence: float, now: float) -> bool:
        if not condition:
            self.onset_time = None
            return False
        if confidence < self.min_confidence:
            return False
        if self.last_fired_time is not None and now - self.last_fired_time < self.cooldown:
            return False
        if self.min_duration <= 0:
            self.last_fired_time = now
            return True
        if self.onset_time is None:
            self.onset_time = now
            return False
        if now - self.onset_time >= self.min_duration:
            self.onset_time      = now
            self.last_fired_time = now
            return True
        return False

    def reset(self) -> None:
        self.onset_time = None
        self.last_fired_time = None


class EventEngine:
    """
    Stateful proctoring event engine.

    Parameters
    ----------
    thresholds : dict
        Loaded from configs/thresholds.yaml.
    """

    def __init__(self, thresholds: Optional[Dict] = None) -> None:
        th = (thresholds or {}).get("events", {})

        def _s(key, dur, cd, conf) -> _State:
            c = th.get(key, {})
            return _State(
                name=key,
                min_duration=c.get("duration_seconds", dur),
                cooldown=c.get("cooldown_seconds", cd),
                min_confidence=c.get("min_confidence", conf),
            )

        self._st: Dict[str, _State] = {
            "gaze_away":         _s("gaze_away",         1.5, 3.0, 0.25),
            "multiple_persons":  _s("multiple_persons",  0.0, 5.0, 0.60),
            "suspicious_object": _s("suspicious_object", 0.0, 4.0, 0.55),
            "object_detected":   _s("object_detected",   0.0, 2.0, 0.40),
            "face_missing":      _s("face_missing",      1.5, 2.0, 1.00),
            "low_visibility":    _s("low_visibility",    2.0, 8.0, 1.00),
        }

        # ── gaze_away config ──────────────────────────────────────────
        ga = th.get("gaze_away", {})
        self._yaw_thresh   = ga.get("yaw_threshold_deg",   12.0)
        self._pitch_thresh = ga.get("pitch_threshold_deg",  9.0)

        # Grace: how many consecutive None-gaze frames before onset resets.
        # At 15 fps, 10 frames ≈ 0.6 s — survives a blink without restarting.
        self._none_grace = ga.get("none_grace_frames", 10)
        self._none_count  = 0      # current consecutive None count

        # Separate state for eyes-hidden path
        eyes_dur = ga.get("eyes_hidden_duration_seconds", 1.5)
        self._eyes_hidden_st = _State(
            name="gaze_away_eyes_hidden",
            min_duration=eyes_dur,
            cooldown=ga.get("cooldown_seconds", 3.0),
            min_confidence=1.0,
        )

        # ── multiple_persons config ───────────────────────────────────
        mp = th.get("multiple_persons", {})
        self._mp_min_conf = mp.get("min_confidence", 0.60)
        self._mp_min_area = mp.get("min_bbox_area", 1536)

        # ── low_visibility config ─────────────────────────────────────
        vis = th.get("low_visibility", {})
        self._brightness_thresh = vis.get("brightness_threshold", 30.0)
        self._blur_thresh       = vis.get("blur_threshold",       12.0)

        # ── suspicious_object config ──────────────────────────────────
        cfg_susp = th.get("suspicious_object", {})
        self._suspicious_names = frozenset(
            cfg_susp.get("classes", list(SUSPICIOUS_NAMES))
        )

        self._events: List[ProctoringEvent] = []

        log.info(
            "EventEngine ready — gaze yaw=%.0f° pitch=%.0f° grace=%d frames | "
            "low_vis bright<%.0f AND sharp<%.0f",
            self._yaw_thresh, self._pitch_thresh, self._none_grace,
            self._brightness_thresh, self._blur_thresh,
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def detect_events(
        self,
        gaze_result:   Optional[tuple],
        object_result: Optional[ObjectResult],
        face_state:    Dict[str, Any],
        frame_index:   int = 0,
    ) -> List[ProctoringEvent]:
        """
        Evaluate all conditions for one frame.

        Parameters
        ----------
        gaze_result : (yaw_deg, pitch_deg) | None
        object_result : ObjectResult | None
        face_state : {num_faces, brightness, sharpness}
        frame_index : int
        """
        now        = time.monotonic()
        fired:     List[ProctoringEvent] = []
        num_faces  = face_state.get("num_faces", 0)
        brightness = face_state.get("brightness", 255.0)
        sharpness  = face_state.get("sharpness",  999.0)

        # ── 1. face_missing ───────────────────────────────────────────
        if self._st["face_missing"].update(num_faces == 0, 1.0, now):
            fired.append(self._evt("face_missing", 1.0, frame_index))

        # ── 2. low_visibility (AND — both conditions must be true) ────
        # OR was causing weak cameras to always trigger on sharpness alone.
        low_vis = (brightness < self._brightness_thresh
                   and sharpness  < self._blur_thresh)
        if self._st["low_visibility"].update(low_vis, 1.0, now):
            fired.append(self._evt(
                "low_visibility", 1.0, frame_index,
                details={
                    "brightness": round(brightness, 1),
                    "sharpness":  round(sharpness,  1),
                },
            ))

        # ── 3. gaze_away ──────────────────────────────────────────────
        if num_faces > 0:
            if gaze_result is not None:
                yaw, pitch = gaze_result
                self._none_count = 0   # valid frame — reset grace counter

                away = (abs(yaw) > self._yaw_thresh
                        or abs(pitch) > self._pitch_thresh)

                # Margin-based confidence: how far past the threshold.
                # A yaw of 24° with threshold 12° → margin = 12 → conf = 0.86.
                # Values near the threshold still produce usable confidence.
                yaw_margin   = max(0.0, abs(yaw)   - self._yaw_thresh)
                pitch_margin = max(0.0, abs(pitch) - self._pitch_thresh)
                gconf = min(
                    max(yaw_margin, pitch_margin) / (self._yaw_thresh * 1.5),
                    1.0,
                ) if away else 0.0

                if self._st["gaze_away"].update(away, gconf, now):
                    fired.append(self._evt(
                        "gaze_away", gconf, frame_index,
                        details={
                            "trigger":   "angle",
                            "yaw_deg":   round(yaw, 2),
                            "pitch_deg": round(pitch, 2),
                        },
                    ))

                # Eyes are visible — reset eyes-hidden timer
                self._eyes_hidden_st.onset_time = None

            else:
                # gaze is None: iris lost (blink / occlusion / head tilt)
                self._none_count += 1

                if self._none_count >= self._none_grace:
                    # Grace period exhausted — genuinely hidden
                    self._st["gaze_away"].onset_time = None   # reset angle path
                    if self._eyes_hidden_st.update(True, 1.0, now):
                        fired.append(self._evt(
                            "gaze_away", 0.90, frame_index,
                            details={"trigger": "eyes_hidden"},
                        ))
                # else: within grace period — preserve onset timer, do nothing
        else:
            # No face at all — reset everything
            self._none_count = 0
            self._st["gaze_away"].onset_time = None
            self._eyes_hidden_st.onset_time  = None

        # ── 4. multiple_persons (with bbox area filter) ───────────────
        if object_result:
            valid_persons = [
                d for d in object_result.detections
                if d.is_person
                and d.confidence >= self._mp_min_conf
                and (d.bbox[2] * d.bbox[3]) >= self._mp_min_area
            ]
            if len(valid_persons) > 1:
                mp_conf = max(d.confidence for d in valid_persons)
                if self._st["multiple_persons"].update(True, mp_conf, now):
                    fired.append(self._evt(
                        "multiple_persons", mp_conf, frame_index,
                        details={"num_persons": len(valid_persons)},
                    ))
            else:
                self._st["multiple_persons"].onset_time = None

        # ── 5. suspicious_object ──────────────────────────────────────
        if object_result and object_result.suspicious_objects:
            for det in object_result.suspicious_objects:
                if det.class_name in self._suspicious_names:
                    if self._st["suspicious_object"].update(True, det.confidence, now):
                        fired.append(self._evt(
                            "suspicious_object", det.confidence, frame_index,
                            bbox=list(det.bbox),
                            details={"object": det.class_name},
                        ))
                    break
        else:
            self._st["suspicious_object"].onset_time = None

        # ── 6. object_detected (generic) ──────────────────────────────
        if object_result:
            generic = [
                d for d in object_result.detections
                if not d.is_person and not d.is_suspicious
            ]
            if generic:
                det = generic[0]
                if self._st["object_detected"].update(True, det.confidence, now):
                    fired.append(self._evt(
                        "object_detected", det.confidence, frame_index,
                        bbox=list(det.bbox),
                        details={"object": det.class_name},
                    ))

        self._events.extend(fired)
        return fired

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @property
    def session_events(self) -> List[ProctoringEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
        for s in self._st.values():
            s.reset()
        self._eyes_hidden_st.reset()
        self._none_count = 0

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _evt(
        event_type: str,
        confidence: float,
        frame_index: int,
        bbox: Optional[List[int]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ProctoringEvent:
        return ProctoringEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            type=event_type,
            confidence=confidence,
            frame_index=frame_index,
            bbox=bbox,
            details=details or {},
        )
