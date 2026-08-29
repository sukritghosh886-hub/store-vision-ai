from __future__ import annotations

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

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
):

    model = get_model()

    result = model.predict(
        frame,
        conf=confidence,
        verbose=False,
    )[0]

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
):

    frame = decode_image(
        image_bytes
    )

    annotated, detections = detect_objects(
        frame,
        confidence,
    )

    return {
        "image": annotated,
        "detections": detections,
        "count": len(detections),
    }