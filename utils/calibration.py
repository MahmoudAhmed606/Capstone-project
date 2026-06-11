"""
Gaze bias calibration — 5-point routine.

The user is shown five reference dots (centre + four corners) one at a
time.  Raw gaze predictions collected at each dot are averaged and
compared to the expected angle derived from the dot's screen position.
The resulting per-axis bias offsets correct for camera placement and
head-pose drift.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _CalibPoint:
    screen_pos: Tuple[float, float]         # (x_ratio, y_ratio) ∈ [0, 1]
    samples: List[Tuple[float, float]] = field(default_factory=list)   # (yaw°, pitch°)


@dataclass
class CalibrationResult:
    """Output from a completed calibration run."""
    yaw_bias:   float = 0.0
    pitch_bias: float = 0.0
    is_valid:   bool  = False
    num_samples: int  = 0


class GazeCalibrator:
    """
    Five-point gaze bias calibration.

    Parameters
    ----------
    duration_per_point:
        Seconds to hold gaze on each dot.
    num_points:
        Number of reference dots (max 5).
    fov_horizontal_deg:
        Assumed horizontal field of view for screen → angle mapping.
    fov_vertical_deg:
        Assumed vertical field of view for screen → angle mapping.
    """

    _REFERENCE_POSITIONS = [
        (0.5, 0.5),   # centre
        (0.1, 0.1),   # top-left
        (0.9, 0.1),   # top-right
        (0.1, 0.9),   # bottom-left
        (0.9, 0.9),   # bottom-right
    ]

    def __init__(
        self,
        duration_per_point: float = 1.5,
        num_points: int = 5,
        fov_horizontal_deg: float = 60.0,
        fov_vertical_deg:   float = 40.0,
    ) -> None:
        self._dur  = duration_per_point
        self._fov_h = fov_horizontal_deg
        self._fov_v = fov_vertical_deg
        n = min(num_points, len(self._REFERENCE_POSITIONS))
        self._points = [_CalibPoint(p) for p in self._REFERENCE_POSITIONS[:n]]
        self._result: Optional[CalibrationResult] = None

    # ------------------------------------------------------------------
    # Interactive mode (OpenCV window)
    # ------------------------------------------------------------------

    def run_interactive(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        gaze_predictor: Callable[[np.ndarray], Optional[Tuple[float, float]]],
        window_name: str = "Calibration",
    ) -> CalibrationResult:
        """
        Display calibration dots on a live camera feed and collect samples.

        Parameters
        ----------
        frame_provider:
            ``() → BGR frame | None``
        gaze_predictor:
            ``(frame) → (yaw_deg, pitch_deg) | None``
        """
        for idx, point in enumerate(self._points):
            log.info("Calibration point %d/%d", idx + 1, len(self._points))
            deadline = time.monotonic() + self._dur

            while time.monotonic() < deadline:
                frame = frame_provider()
                if frame is None:
                    continue

                display = frame.copy()
                self._draw_overlay(
                    display, point.screen_pos,
                    idx, len(self._points),
                    progress=(self._dur - (deadline - time.monotonic())) / self._dur,
                )
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.warning("Calibration aborted.")
                    return CalibrationResult(is_valid=False)

                pred = gaze_predictor(frame)
                if pred is not None:
                    point.samples.append(pred)

            log.debug("Point %d: %d samples", idx + 1, len(point.samples))

        cv2.destroyWindow(window_name)
        self._result = self._compute_bias()
        return self._result

    # ------------------------------------------------------------------
    # Headless mode (for testing / no-display environments)
    # ------------------------------------------------------------------

    def feed_sample(self, point_index: int, yaw_deg: float, pitch_deg: float) -> None:
        """Feed one gaze sample to a calibration point (headless)."""
        if 0 <= point_index < len(self._points):
            self._points[point_index].samples.append((yaw_deg, pitch_deg))

    def finalize(self) -> CalibrationResult:
        """Compute bias from accumulated samples (headless mode)."""
        self._result = self._compute_bias()
        return self._result

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
        """Apply bias correction to a raw gaze prediction."""
        if self._result is None or not self._result.is_valid:
            return yaw_deg, pitch_deg
        return yaw_deg - self._result.yaw_bias, pitch_deg - self._result.pitch_bias

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_bias(self) -> CalibrationResult:
        yaw_errs, pitch_errs = [], []
        for point in self._points:
            if not point.samples:
                continue
            arr        = np.array(point.samples)
            pred_yaw   = float(arr[:, 0].mean())
            pred_pitch = float(arr[:, 1].mean())
            exp_yaw, exp_pitch = self._screen_to_angle(*point.screen_pos)
            yaw_errs.append(pred_yaw   - exp_yaw)
            pitch_errs.append(pred_pitch - exp_pitch)

        if not yaw_errs:
            log.warning("No calibration samples — bias set to zero.")
            return CalibrationResult(is_valid=False)

        bias_yaw   = float(np.mean(yaw_errs))
        bias_pitch = float(np.mean(pitch_errs))
        total      = sum(len(p.samples) for p in self._points)
        log.info(
            "Calibration done — yaw_bias=%.2f°  pitch_bias=%.2f°  samples=%d",
            bias_yaw, bias_pitch, total,
        )
        return CalibrationResult(
            yaw_bias=bias_yaw, pitch_bias=bias_pitch,
            is_valid=True, num_samples=total,
        )

    def _screen_to_angle(self, x_ratio: float, y_ratio: float) -> Tuple[float, float]:
        """Map screen position fraction → expected gaze angles in degrees."""
        yaw   = (x_ratio - 0.5) * self._fov_h
        pitch = (0.5 - y_ratio) * self._fov_v
        return yaw, pitch

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        pos: Tuple[float, float],
        step: int,
        total: int,
        progress: float = 0.0,
    ) -> None:
        h, w = frame.shape[:2]
        cx, cy = int(pos[0] * w), int(pos[1] * h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.circle(frame, (cx, cy), 18, (0, 220, 0), -1)
        cv2.circle(frame, (cx, cy), 22, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Look at the green dot  ({step + 1}/{total})",
            (w // 2 - 155, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )
        # Progress bar
        filled = int(progress * w)
        cv2.rectangle(frame, (0, h - 6), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 6), (filled, h), (0, 200, 0), -1)