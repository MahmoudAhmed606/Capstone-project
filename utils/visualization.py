"""
Frame annotation and real-time visualization helpers.

Changes
-------
- draw_face_box(): new function — draws the face bounding box with a
  corner-bracket style (looks cleaner than a full rectangle).
- draw_gaze_vector(): arrow is longer (120 px), thicker (3 px), and
  colour-coded: green = looking at screen, orange = slight deviation,
  red = gaze_away territory. Shows yaw/pitch values as text beside arrow.
- draw_event_panel(): unchanged interface, minor style polish.
- frame_quality_check(): unchanged.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Colour palette (BGR) ───────────────────────────────────────────────────
C_OK    = (50,  205,  50)     # green      — looking at screen
C_WARN  = (0,   165, 255)     # orange     — slight deviation
C_ALERT = (30,   30, 220)     # red        — gaze_away territory
C_FACE  = (255, 200,   0)     # gold       — face box
C_INFO  = (200, 200, 200)     # light grey — text


def draw_face_box(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    color: Tuple = C_FACE,
) -> None:
    """
    Draw a corner-bracket style face bounding box.

    The brackets occupy the corners only (not a full rectangle) which
    looks cleaner over a busy background and is less distracting.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return

    arm = max(12, min(w, h) // 5)   # bracket arm length (adaptive)
    t   = 2                          # line thickness

    # Top-left
    cv2.line(frame, (x, y),         (x + arm, y),         color, t)
    cv2.line(frame, (x, y),         (x, y + arm),         color, t)
    # Top-right
    cv2.line(frame, (x + w, y),     (x + w - arm, y),     color, t)
    cv2.line(frame, (x + w, y),     (x + w, y + arm),     color, t)
    # Bottom-left
    cv2.line(frame, (x, y + h),     (x + arm, y + h),     color, t)
    cv2.line(frame, (x, y + h),     (x, y + h - arm),     color, t)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - arm, y + h), color, t)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - arm), color, t)


def draw_landmarks(
    frame:        np.ndarray,
    landmarks_px: np.ndarray,
    color:        Tuple = C_FACE,
    radius:       int   = 1,
) -> None:
    """Draw 2-D face landmarks in-place."""
    for pt in landmarks_px:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)


def draw_gaze_vector(
    frame:       np.ndarray,
    origin:      Tuple[int, int],
    yaw_deg:     float,
    pitch_deg:   float,
    length:      int   = 120,
    yaw_thresh:  float = 12.0,
    pitch_thresh: float = 9.0,
) -> None:
    """
    Draw a gaze direction arrow from ``origin``.

    The arrow is colour-coded by how far gaze deviates from centre:
      Green  — within threshold (looking at screen)
      Orange — 50–100 % of threshold (slight deviation)
      Red    — beyond threshold (gaze_away territory)

    Also draws the yaw/pitch values as small text beside the arrowhead.

    Parameters
    ----------
    origin : (x, y)
        Start point — should be nose tip or face bbox centre.
    yaw_deg : float
        Positive = looking right.
    pitch_deg : float
        Positive = looking up.
    length : int
        Arrow length in pixels at the original display resolution.
    yaw_thresh, pitch_thresh : float
        Same values as in thresholds.yaml — used for colour coding.
    """
    # Arrow direction: yaw = horizontal, pitch = vertical (inverted Y axis)
    dx = int(length * math.sin(math.radians(yaw_deg)))
    dy = int(-length * math.sin(math.radians(pitch_deg)))

    x0, y0 = origin
    x1, y1 = x0 + dx, y0 + dy

    # Colour coding based on deviation from threshold
    yaw_ratio   = abs(yaw_deg)   / (yaw_thresh   + 1e-6)
    pitch_ratio = abs(pitch_deg) / (pitch_thresh  + 1e-6)
    ratio       = max(yaw_ratio, pitch_ratio)

    if ratio < 0.5:
        color = C_OK
    elif ratio < 1.0:
        color = C_WARN
    else:
        color = C_ALERT

    # Arrow shaft + head
    cv2.arrowedLine(frame, (x0, y0), (x1, y1), color, 3, tipLength=0.28)

    # Small dot at origin so it's visible when yaw/pitch are near zero
    cv2.circle(frame, (x0, y0), 4, color, -1)

    # Yaw / pitch readout beside arrowhead
    label = f"y:{yaw_deg:+.0f} p:{pitch_deg:+.0f}"
    tx    = x1 + 6 if x1 + 6 + 100 < frame.shape[1] else x1 - 110
    ty    = y1 - 4
    cv2.putText(
        frame, label, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
    )


def draw_bbox(
    frame:     np.ndarray,
    bbox:      Tuple[int, int, int, int],
    label:     str   = "",
    color:     Tuple = C_OK,
    thickness: int   = 2,
) -> None:
    """Draw a bounding box with an optional label tag."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA,
        )


def draw_event_panel(
    frame:         np.ndarray,
    active_events: List[str],
    fps:           float = 0.0,
) -> None:
    """
    Draw the heads-up event status panel (top-left corner).

    Active event → red dot.   Inactive → dim grey dot.
    """
    ALL_EVENTS = [
        "gaze_away", "multiple_persons", "suspicious_object",
        "face_missing", "low_visibility",
    ]
    panel_w = 230
    panel_h = 22 + len(ALL_EVENTS) * 22 + 10
    x0, y0  = 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (75, 75, 75), 1)

    cv2.putText(
        frame, f"ProctorVision  {fps:.1f} fps",
        (x0 + 8, y0 + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_INFO, 1, cv2.LINE_AA,
    )

    for i, ev in enumerate(ALL_EVENTS):
        ey     = y0 + 24 + i * 22
        active = ev in active_events
        dot_c  = C_ALERT if active else (55, 55, 55)
        txt_c  = C_ALERT if active else (90, 90, 90)
        cv2.circle(frame, (x0 + 14, ey + 6), 5, dot_c, -1)
        cv2.putText(
            frame, ev, (x0 + 26, ey + 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_c, 1, cv2.LINE_AA,
        )


def frame_quality_check(frame: np.ndarray) -> Tuple[float, float]:
    """
    Return (brightness, sharpness) for low-visibility detection.

    brightness — mean grey pixel value 0–255.
    sharpness  — Laplacian variance (higher = sharper).

    Must be called on the ORIGINAL full-res frame, NOT a downscaled copy.
    Bilinear resize reduces high-frequency content and artificially lowers
    the sharpness score, causing false low_visibility events.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
