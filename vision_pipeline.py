from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

PERSON_CLASS = 0

# COCO classes that are useful as generic retail-object detections.
COCO_ITEMS = {
    24: "backpack",
    26: "handbag",
    39: "bottle",
    41: "cup",
    45: "bowl",
    46: "banana",
    47: "apple",
    49: "orange",
    67: "cell_phone",
}


_model = None


def get_model():

    global _model

    if _model is None:
        _model = YOLO(MODEL_NAME)

    return _model


def _bytes_to_bgr(image_bytes: bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rgb = np.array(image)

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def process_image(
    image: Any,
    confidence: float = 0.40,
    input_is_frame: bool = False,
):

    if input_is_frame:

        frame = image.copy()

    else:

        frame = _bytes_to_bgr(
            image
        )

    model = get_model()

    results = model.predict(
        frame,
        conf=confidence,
        verbose=False,
    )

    detections = []

    output = frame.copy()

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(
                box.cls[0].item()
            )

            conf = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            if cls == PERSON_CLASS:

                label = "person"

            elif cls in COCO_ITEMS:

                label = COCO_ITEMS[cls]

            else:

                continue

            detections.append(
                {
                    "label": label,
                    "class_id": cls,
                    "confidence": round(
                        conf,
                        3,
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

            # Draw bounding box.
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 180, 255),
                2,
            )

            cv2.putText(
                output,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
            )

    return {
        "image": output,
        "detections": detections,
    }