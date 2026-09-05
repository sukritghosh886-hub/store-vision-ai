from __future__ import annotations

import io
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from inference_backend import get_backend


MODEL_NAME = os.getenv("STORE_VISION_MODEL", "yolov8n.pt")

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    global _model

    if _model is None:
        _model = YOLO(MODEL_NAME)

    return _model


def decode_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.asarray(image)

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def detect_objects(
    frame: np.ndarray,
    confidence: float = 0.35,
    mode: str = "auto",
):
    """
    Hardware-adaptive object detection.

    auto:
        GPU when CUDA is available, otherwise CPU.

    cpu:
        Lightweight CPU inference.

    gpu:
        GPU inference when available, otherwise safely falls
        back to CPU.
    """

    backend = get_backend(
        mode=mode,
        model_path=MODEL_NAME,
    )

    results = backend.predict(
        frame,
        confidence=confidence,
    )

    result = results[0]

    annotated = frame.copy()

    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:

        cls_id = int(
            box.cls.item()
        )

        score = float(
            box.conf.item()
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist(),
        )

        label = result.names[
            cls_id
        ]

        detections.append(
            {
                "class_id": cls_id,
                "label": label,
                "confidence": score,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 180, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )

    return annotated, detections


def process_image(
    image_bytes: bytes,
    confidence: float = 0.35,
    mode: str = "auto",
):
    frame = decode_image(
        image_bytes
    )

    annotated, detections = detect_objects(
        frame,
        confidence,
        mode,
    )

    return {
        "image": annotated,
        "detections": detections,
        "count": len(detections),
    }


def process_video(
    video_path: str,
    store_id: str,
    shelf_zone_frac: float = 0.55,
    exit_zone_frac: float = 0.85,
    conf: float = 0.40,
    frame_stride: int = 2,
    mode: str = "auto",
):
    """
    Process a video using the hardware-adaptive backend.

    This is intentionally a generator so the FastAPI endpoint can
    process long videos without storing every frame in memory.

    CPU mode automatically uses a larger frame stride unless the
    caller explicitly supplies one.
    """

    backend = get_backend(
        mode=mode,
        model_path=MODEL_NAME,
    )

    if frame_stride is None or frame_stride < 1:
        frame_stride = backend.default_frame_stride

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_number = 0
    inference_count = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            # Frame sampling reduces CPU workload.
            if (
                (frame_number - 1)
                % frame_stride
                != 0
            ):
                continue

            annotated, detections = detect_objects(
                frame,
                confidence=conf,
                mode=backend.mode,
            )

            inference_count += 1

            people = sum(
                1
                for item in detections
                if item["label"].lower()
                == "person"
            )

            message = (
                f"frame={frame_number} "
                f"detections={len(detections)} "
                f"people={people} "
                f"mode={backend.mode}"
            )

            yield annotated, message

    finally:
        cap.release()