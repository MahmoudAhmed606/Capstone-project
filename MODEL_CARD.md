# ProctorVision Model Card

## Models used

### 1. MediaPipe Face Landmarker
- **Source:** Google MediaPipe (Apache 2.0)
- **Task:** 478-point face landmark detection + iris tracking
- **Input:** RGB image, any resolution
- **Output:** Normalised (x, y, z) per landmark, presence/visibility scores
- **Limitations:** Accuracy drops with extreme head poses (> 60°), heavy occlusion, very dark frames

### 2. L2CS-Net (ResNet-50)
- **Source:** https://github.com/Ahmednull/L2CS-Net (MIT)
- **Training data:** Gaze360 (60 K+ images, 360° gaze in the wild)
- **Task:** Gaze estimation — yaw + pitch as bin-classification expected value
- **Input:** 448×448 face crop, ImageNet normalised
- **Output:** yaw_deg, pitch_deg ∈ [−45°, +45°]
- **Limitations:** Assumes frontal-ish face; accuracy degrades > ±40°; not trained on classroom/webcam domain — fine-tuning recommended

### 3. YOLO11n (baseline) / YOLO26n (target)
- **Source:** Ultralytics (AGPL-3.0)
- **Training data (base):** COCO 2017
- **Fine-tune target:** person, mobile, paper, calculator, book, earphones, laptop
- **Input:** 640×640 BGR
- **Output:** Class-ID, confidence, xyxy bounding box
- **Limitations:** Base COCO model does not distinguish mobile/earphones/paper as separate classes — fine-tuning on the proctoring dataset is required for full detection coverage

## Intended use

- Exam proctoring on candidate machines
- Local inference only — no video transmitted externally
- Not intended for law enforcement, biometric identification, or surveillance

## Out-of-scope use

- Identification of individuals by face (system does not perform face recognition)
- Continuous workplace monitoring without informed consent
- Any deployment where GDPR / FERPA / CCPA consent has not been obtained

## Bias and fairness considerations

- L2CS-Net was trained on Gaze360, which has limited representation of non-frontal or
  profile-facing subjects; gaze accuracy may be lower for extreme poses
- YOLO base model (COCO) shows variable performance on small objects; suspicious items
  such as earphones require fine-tuning for reliable detection
- Webcam quality, lighting, and skin-tone variation affect landmark detection accuracy

## Privacy

- Raw frames are never stored when `PRIVACY_MODE=true`
- Violation snapshots (when enabled) are downscaled to ≤ 320 px and stored locally only
- No personally identifiable information is transmitted outside the local machine