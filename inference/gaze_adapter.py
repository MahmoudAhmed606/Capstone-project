"""
L2CS-Net gaze estimation adapter — eye-movement-sensitive heuristic.

Heuristic rewrite
-----------------
Previous issue: normalization used full eye-corner-to-corner span as
denominator. A student looking sideways with eyes only (no head turn)
moves the iris ~20-25% of that span → maps to only 8-10°, which was
below the 12° threshold → gaze_away never fired.

Fix: normalize by HALF-span (eye centre to corner). Same 20% iris
movement now maps to 20-22°, well above the 10° threshold. The scale
factor is also raised from 40 → 55 to spread the usable angle range.

Three detection layers (in order):
  1. Eye-openness check  — if eyes hidden, return None → eyes_hidden timer
  2. Iris-offset heuristic — eye-movement-sensitive, half-span normalised
  3. Profile detection   — catches large head turns independently
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

log = logging.getLogger(__name__)

NUM_BINS    = 90
ANGLE_RANGE = 90

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_BIN_CENTERS: np.ndarray = np.linspace(
    -ANGLE_RANGE / 2, ANGLE_RANGE / 2, NUM_BINS, dtype=np.float32
)

# Eyes-closed threshold: vertical lid gap / face height.
# Below this value both eyes are considered closed/hidden.
_EYE_OPEN_MIN = 0.012


# ── Model (layer names must match L2CSNet_gaze360.pkl) ─────────────────────

class L2CSNet(nn.Module):
    def __init__(self, num_bins: int = NUM_BINS) -> None:
        super().__init__()
        r = tv_models.resnet50(weights=None)
        self.conv1   = r.conv1;  self.bn1     = r.bn1
        self.relu    = r.relu;   self.maxpool = r.maxpool
        self.layer1  = r.layer1; self.layer2  = r.layer2
        self.layer3  = r.layer3; self.layer4  = r.layer4
        self.avgpool = r.avgpool
        self.fc_yaw   = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc_yaw(x), self.fc_pitch(x)


def _logits_to_angle(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return float((probs * _BIN_CENTERS).sum())


# ── Adapter ────────────────────────────────────────────────────────────────

class GazeAdapter:
    """
    Gaze estimation — neural net, ONNX, or eye-movement-sensitive heuristic.

    Parameters
    ----------
    model_path : str
    device : str        'cpu' or 'cuda'
    use_onnx : bool
    use_heuristic : bool
        Fast iris-geometry (~0 ms). After this fix, correctly detects
        eye-only gaze changes without requiring a head turn.
    input_size : int    crop resolution for neural net (ignored in heuristic)
    """

    def __init__(
        self,
        model_path:    str  = "models/L2CSNet_gaze360.pkl",
        device:        str  = "cpu",
        use_onnx:      bool = False,
        use_heuristic: bool = True,
        input_size:    int  = 448,
    ) -> None:
        self.device         = torch.device(device)
        self._model: Optional[L2CSNet] = None
        self._ort           = None
        self._use_onnx      = False
        self._use_heuristic = use_heuristic
        self._input_size    = input_size

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        if use_heuristic:
            log.info("GazeAdapter: eye-movement-sensitive heuristic mode.")
            return

        p = Path(model_path)
        if use_onnx or p.suffix == ".onnx":
            self._load_onnx(str(p))
        elif p.suffix in (".pt", ".pth", ".pkl") and p.exists():
            self._load_pytorch(str(p))
        else:
            log.warning("Weights not found at '%s' — using heuristic.", model_path)
            self._use_heuristic = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_gaze(
        self,
        face_crop_bgr: Optional[np.ndarray],
        landmarks_px:  Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Return (yaw_deg, pitch_deg) or None.

        None means eyes are closed / iris hidden. The event engine's
        eyes-hidden timer handles this case separately.
        """
        try:
            if self._use_heuristic:
                return self._heuristic(landmarks_px)
            if face_crop_bgr is None or face_crop_bgr.size == 0:
                return None
            return (self._infer_onnx(face_crop_bgr) if self._use_onnx
                    else self._infer_pytorch(face_crop_bgr))
        except Exception as exc:
            log.debug("Gaze inference error: %s", exc)
            return None

    def export_onnx(self, output_path: str = "models/l2cs_net.onnx") -> None:
        if self._model is None:
            raise RuntimeError("No PyTorch model loaded.")
        dummy = torch.zeros(1, 3, self._input_size, self._input_size,
                            device=self.device)
        torch.onnx.export(self._model, dummy, output_path,
                          input_names=["face"],
                          output_names=["yaw_logits", "pitch_logits"],
                          opset_version=17,
                          dynamic_axes={"face": {0: "batch"}})
        log.info("ONNX exported → %s", output_path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_pytorch(self, path: str) -> None:
        self._model = L2CSNet(num_bins=NUM_BINS)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = (ckpt.get("model_state_dict") or ckpt.get("state_dict")
                  or ckpt.get("model") or ckpt)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, _ = self._model.load_state_dict(sd, strict=False)
        if any("fc_yaw" in k or "fc_pitch" in k for k in missing):
            log.error("CRITICAL: fc_yaw/fc_pitch not loaded!")
        self._model.eval().to(self.device)
        log.info("L2CS-Net loaded — %d/%d keys.", len(sd) - len(missing), len(sd))

    def _load_onnx(self, path: str) -> None:
        if not Path(path).exists():
            self._use_heuristic = True
            return
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort      = ort.InferenceSession(path, sess_options=opts,
                                              providers=["CPUExecutionProvider"])
        self._use_onnx = True

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _infer_pytorch(self, bgr: np.ndarray) -> Tuple[float, float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y, p = self._model(inp)
        return _logits_to_angle(y[0]), _logits_to_angle(p[0])

    def _infer_onnx(self, bgr: np.ndarray) -> Tuple[float, float]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).numpy()
        yo, po = self._ort.run(None, {"face": inp})
        return _logits_to_angle(torch.tensor(yo[0])), _logits_to_angle(torch.tensor(po[0]))

    # ------------------------------------------------------------------
    # Heuristic — three-layer iris geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic(lm: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """
        Eye-movement-sensitive gaze heuristic.

        Layer 1 — Eye openness guard
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Returns None if both lids are nearly closed.

        Layer 2 — Iris offset (primary signal, detects eye-only gaze)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Measures how far the iris centre sits from the eye horizontal
        centre, normalised by HALF the eye width (centre to corner).

        Why half-span matters
        ---------------------
        Full-span normalisation (old code):
          iris at 20% from eye-centre → t ≈ 0.20 → yaw ≈ 0.20 × 40 = 8°
          (below 10° threshold → missed)

        Half-span normalisation (new code):
          iris at 20% from eye-centre → dev ≈ 0.40 → yaw ≈ 0.40 × 55 = 22°
          (above 10° threshold → detected ✓)

        Layer 3 — Profile detection (catches large head turns)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Nose-to-ear asymmetry fires for significant head rotation
        independently of iris tracking.
        """
        if lm is None or len(lm) < 478:
            return None

        pts = lm.astype(float)

        # ── Layer 1: eye openness ──────────────────────────────────────
        face_h = max(pts[:, 1].max() - pts[:, 1].min(), 1.0)
        l_open = abs(pts[145][1] - pts[159][1]) / face_h   # lower - upper lid
        r_open = abs(pts[374][1] - pts[386][1]) / face_h
        if (l_open + r_open) / 2.0 < _EYE_OPEN_MIN:
            return None   # both eyes closed/hidden → eyes-hidden timer fires

        # ── Layer 2: iris half-span offset ─────────────────────────────
        # Iris centres
        l_iris = pts[468];  r_iris = pts[473]

        # Eye corners
        l_out = pts[33];   l_in  = pts[133]
        r_in  = pts[362];  r_out = pts[263]

        # Eye horizontal centre (midpoint of two corners)
        l_cx = (l_out[0] + l_in[0]) / 2.0
        r_cx = (r_in[0]  + r_out[0]) / 2.0

        # Half-span (centre → corner distance)
        l_hs = max(abs(l_in[0] - l_out[0]) / 2.0, 1.0)
        r_hs = max(abs(r_out[0] - r_in[0]) / 2.0, 1.0)

        # Normalised iris deviation in [-1, 1]:
        #  left eye:  positive = iris toward inner corner = looking right
        #  right eye: positive = iris toward outer corner = looking right
        l_dev = (l_iris[0] - l_cx) / l_hs
        r_dev = (r_iris[0] - r_cx) / r_hs

        yaw_raw = (l_dev + r_dev) / 2.0
        # Scale factor 55: comfortable "looking away" (35% of half-span)
        # → dev ≈ 0.35 × 2 = 0.70 → yaw ≈ 0.35 × 55 = 19° (above 10° threshold)
        yaw_deg = float(np.clip(yaw_raw * 55.0, -45.0, 45.0))

        # Pitch: iris vertical offset within lid gap
        l_mid_y = (pts[159][1] + pts[145][1]) / 2.0
        l_lid_h  = max(abs(pts[145][1] - pts[159][1]), 1.0)
        pitch_raw = -((l_iris[1] - l_mid_y) / l_lid_h)   # positive = up
        pitch_deg = float(np.clip(pitch_raw * 40.0, -30.0, 30.0))

        # ── Layer 3: profile / large head-turn detection ───────────────
        # Nose tip (1) vs ear tragus landmarks (234 left, 454 right)
        nose  = pts[1]
        l_ear = pts[234];  r_ear = pts[454]
        face_w = max(pts[:, 0].max() - pts[:, 0].min(), 1.0)

        # Horizontal distance from nose to each ear in screen space.
        # When face turns right: right ear disappears, left ear gets closer.
        l_ear_d = abs(nose[0] - l_ear[0])
        r_ear_d = abs(nose[0] - r_ear[0])
        ear_asym = (r_ear_d - l_ear_d) / face_w   # positive = turning right

        if abs(ear_asym) > 0.22:
            # Large head-turn: override with profile-based yaw
            profile_yaw = float(np.clip(ear_asym * 85.0, -45.0, 45.0))
            # Use whichever signal is stronger (profile or iris)
            if abs(profile_yaw) > abs(yaw_deg):
                yaw_deg = profile_yaw

        return yaw_deg, pitch_deg