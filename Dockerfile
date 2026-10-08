FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN python - <<'PY'
import cv2
import torch
import torchvision
import fastapi
import ultralytics
import supabase

print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("FastAPI:", fastapi.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Supabase: IMPORT PASS")
print("RUNTIME_IMPORT_TEST=PASS")
PY

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8008}"]