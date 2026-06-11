# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="proctor-vision"
LABEL description="ProctorVision — local AI proctoring inference"

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ────────────────────────────────────────────────────────────
COPY . .

# Create output directory
RUN mkdir -p outputs models datasets

# ── Model download helper (optional — mount weights instead) ───────────────
# Uncomment to bake YOLO nano weights into the image (~6 MB):
# RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# ── Expose API port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Default command: API server ────────────────────────────────────────────
# Override with:  docker run ... python examples/run_video.py --video /data/exam.mp4
CMD ["uvicorn", "inference.api_server:app", "--host", "0.0.0.0", "--port", "8000"]