"""
Store Vision AI - Multi-Instance Object Detection Engine

Designed for:
- Multiple copies of the same product
- Overlapping products
- Books
- Bottles
- Caps
- Stickers
- Other store products

Recommended model:
    A CUSTOM YOLO INSTANCE-SEGMENTATION MODEL

Example:
    models/store_products_seg.pt
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "STORE_VISION_MODEL",
    "models/store_products_seg.pt"
)

CPU_IMGSZ = int(
    os.getenv(
        "STORE_VISION_CPU_IMGSZ",
        "960"
    )
)

GPU_IMGSZ = int(
    os.getenv(
        "STORE_VISION_GPU_IMGSZ",
        "1280"
    )
)

DEFAULT_CONFIDENCE = float(
    os.getenv(
        "STORE_VISION_CONFIDENCE",
        "0.25"
    )
)

DEFAULT_IOU = float(
    os.getenv(
        "STORE_VISION_IOU",
        "0.50"
    )
)

# IMPORTANT:
# Increase this so 27, 50, 100+ objects can be returned.
MAX_DETECTIONS = int(
    os.getenv(
        "STORE_VISION_MAX_DETECTIONS",
        "1000"
    )
)


# ============================================================
# MODEL CACHE
# ============================================================

_models: Dict[str, YOLO] = {}


# ============================================================
# HARDWARE
# ============================================================

def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_device() -> str:
    return "cuda:0" if cuda_available() else "cpu"


# ============================================================
# MODEL
# ============================================================

def get_model() -> YOLO:
    """
    Load the custom segmentation model only once.
    """

    device = get_device()

    if device in _models:
        return _models[device]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Store Vision AI model not found: {MODEL_PATH}\n"
            "Train/export your custom segmentation model first."
        )

    print(
        f"Loading Store Vision AI segmentation model: "
        f"{MODEL_PATH}"
    )

    model = YOLO(MODEL_PATH)

    _models[device] = model

    return model


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(
    image_bytes: bytes
) -> np.ndarray:

    if not image_bytes:
        raise ValueError(
            "No image data was provided."
        )

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError(
            "Could not decode uploaded image."
        )

    return frame


# ============================================================
# MASK AREA
# ============================================================

def calculate_mask_area(
    mask: np.ndarray
) -> int:

    return int(
        np.count_nonzero(mask)
    )


# ============================================================
# CENTROID
# ============================================================

def calculate_centroid(
    bbox: List[float]
) -> List[float]:

    x1, y1, x2, y2 = bbox

    return [
        float((x1 + x2) / 2),
        float((y1 + y2) / 2)
    ]


# ============================================================
# DETECTION
# ============================================================

def detect_objects(
    frame: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    profile: str | None = None,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:

    if frame is None:
        raise ValueError(
            "Image frame is empty."
        )

    if not isinstance(frame, np.ndarray):
        raise ValueError(
            "frame must be a numpy array."
        )

    if frame.size == 0:
        raise ValueError(
            "Image frame contains no data."
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    if not 0.0 < iou < 1.0:
        raise ValueError(
            "IoU must be between 0 and 1."
        )

    model = get_model()

    device = get_device()

    imgsz = (
        GPU_IMGSZ
        if device.startswith("cuda")
        else CPU_IMGSZ
    )

    predict_kwargs = {
        "source": frame,

        "conf": confidence,

        "iou": iou,

        "imgsz": imgsz,

        "device": device,

        "max_det": MAX_DETECTIONS,

        "verbose": False,

        # Test-time augmentation can improve
        # difficult/partially occluded detections.
        "augment": True,

        # Do NOT merge different instances.
        "agnostic_nms": False,
    }

    if device.startswith("cuda"):
        predict_kwargs["half"] = True

    results = model.predict(
        **predict_kwargs
    )

    if not results:
        return frame.copy(), []

    result = results[0]

    # ========================================================
    # DRAW SEGMENTATION RESULT
    # ========================================================

    annotated = result.plot(
        boxes=True,
        masks=True,
        labels=True,
        conf=True
    )

    detections: List[Dict[str, Any]] = []

    names = result.names

    # ========================================================
    # INSTANCE SEGMENTATION
    # ========================================================

    if result.boxes is None:
        return annotated, detections

    masks = result.masks

    for index, box in enumerate(result.boxes):

        class_id = int(
            box.cls[0].item()
        )

        conf = float(
            box.conf[0].item()
        )

        bbox = [
            float(value)
            for value
            in box.xyxy[0].tolist()
        ]

        class_name = str(
            names[class_id]
        )

        centroid = calculate_centroid(
            bbox
        )

        mask_area = None

        polygon = None

        if masks is not None:

            # Binary mask for this individual object
            mask = masks.data[index]

            mask = (
                mask
                .detach()
                .cpu()
                .numpy()
            )

            mask = (
                mask > 0.5
            ).astype(np.uint8)

            mask_area = calculate_mask_area(
                mask
            )

            # Polygon coordinates
            if index < len(masks.xy):

                polygon = (
                    masks.xy[index]
                    .tolist()
                )

        detections.append({

            # Unique instance inside this frame
            "instance_id":
                index,

            "class_id":
                class_id,

            "class_name":
                class_name,

            "confidence":
                conf,

            "bbox":
                bbox,

            "centroid":
                centroid,

            "mask_area":
                mask_area,

            "polygon":
                polygon,

        })

    return annotated, detections


# ============================================================
# CLASS COUNTS
# ============================================================

def count_by_class(
    detections: List[Dict[str, Any]]
) -> Dict[str, int]:

    counts: Dict[str, int] = {}

    for detection in detections:

        class_name = detection[
            "class_name"
        ]

        counts[class_name] = (
            counts.get(class_name, 0)
            + 1
        )

    return counts


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

def calculate_bbox_iou(
    box_a: List[float],
    box_b: List[float]
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1
    )

    intersection_height = max(
        0.0,
        iy2 - iy1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def find_overlapping_objects(
    detections: List[Dict[str, Any]],
    threshold: float = 0.20
) -> List[Dict[str, Any]]:

    overlaps = []

    for i in range(
        len(detections)
    ):

        for j in range(
            i + 1,
            len(detections)
        ):

            first = detections[i]
            second = detections[j]

            iou = calculate_bbox_iou(
                first["bbox"],
                second["bbox"]
            )

            if iou >= threshold:

                overlaps.append({

                    "first_instance":
                        first["instance_id"],

                    "second_instance":
                        second["instance_id"],

                    "first_class":
                        first["class_name"],

                    "second_class":
                        second["class_name"],

                    "overlap":
                        round(iou, 4),

                })

    return overlaps


# ============================================================
# HIGH-LEVEL IMAGE PROCESSING
# ============================================================

def process_image(
    image_bytes: bytes,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    profile: str | None = None,
) -> Dict[str, Any]:

    frame = decode_image(
        image_bytes
    )

    annotated, detections = detect_objects(

        frame,

        confidence=confidence,

        iou=iou,

        profile=profile,
    )

    class_counts = count_by_class(
        detections
    )

    overlaps = find_overlapping_objects(
        detections
    )

    height, width = frame.shape[:2]

    return {

        "image":
            annotated,

        # Every individual object
        "detections":
            detections,

        # Total number of instances
        "count":
            len(detections),

        # Example:
        # {
        #     "book": 27,
        #     "bottle": 8,
        #     "cap": 12
        # }
        "class_counts":
            class_counts,

        # Overlap information
        "overlaps":
            overlaps,

        "overlap_count":
            len(overlaps),

        "model":
            MODEL_PATH,

        "device":
            get_device(),

        "imgsz":
            (
                GPU_IMGSZ
                if cuda_available()
                else CPU_IMGSZ
            ),

        "image_size":
            (width, height),

    }


# ============================================================
# VIDEO FRAME PROCESSING
# ============================================================

def process_frame(
    frame: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> Dict[str, Any]:

    annotated, detections = detect_objects(

        frame,

        confidence=confidence,

        iou=iou,
    )

    return {

        "frame":
            annotated,

        "detections":
            detections,

        "count":
            len(detections),

        "class_counts":
            count_by_class(
                detections
            ),

        "overlaps":
            find_overlapping_objects(
                detections
            ),

    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [

    "cuda_available",

    "get_device",

    "get_model",

    "decode_image",

    "detect_objects",

    "count_by_class",

    "find_overlapping_objects",

    "calculate_bbox_iou",

    "process_image",

    "process_frame",

]