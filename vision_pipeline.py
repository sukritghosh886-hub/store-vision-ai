"""
Store Vision AI - Vision Inference Engine

Performance profiles:

AUTO
    Automatically uses FULL_GPU when CUDA is available.
    Otherwise uses LIGHT_CPU.

FULL_GPU
    High-performance YOLO model + high resolution + CUDA.

LIGHT_CPU
    Lightweight YOLO model + lower resolution + CPU.

The GPU path is preserved and is NOT downgraded.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# MODEL CONFIGURATION
# ============================================================

FULL_MODEL_NAME = os.getenv(
    "STORE_VISION_GPU_MODEL",
    "yolov8m.pt"
)

LIGHT_MODEL_NAME = os.getenv(
    "STORE_VISION_CPU_MODEL",
    "yolov8n.pt"
)

FULL_IMGSZ = int(
    os.getenv(
        "STORE_VISION_GPU_IMGSZ",
        "1280"
    )
)

LIGHT_IMGSZ = int(
    os.getenv(
        "STORE_VISION_CPU_IMGSZ",
        "640"
    )
)

DEFAULT_PROFILE = os.getenv(
    "STORE_VISION_PROFILE",
    "auto"
).strip().lower()


VALID_PROFILES = {
    "auto",
    "full_gpu",
    "light_cpu",
}


# Backward compatibility
MODEL_NAME = LIGHT_MODEL_NAME


# Cache loaded models
_models: Dict[str, YOLO] = {}


# ============================================================
# HARDWARE DETECTION
# ============================================================

def cuda_available() -> bool:
    """
    Check whether CUDA GPU is available.
    """
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ============================================================
# PROFILE MANAGEMENT
# ============================================================

def resolve_profile(
    profile: str | None = None
) -> str:
    """
    Resolve requested profile.

    AUTO:
        GPU available -> FULL_GPU
        GPU unavailable -> LIGHT_CPU
    """

    requested = (
        profile or DEFAULT_PROFILE
    ).strip().lower()

    if requested not in VALID_PROFILES:
        raise ValueError(
            f"Unknown profile '{requested}'. "
            f"Choose: auto, full_gpu, light_cpu."
        )

    # Automatic hardware selection
    if requested == "auto":

        if cuda_available():
            return "full_gpu"

        return "light_cpu"

    # Explicit GPU request
    if requested == "full_gpu":

        if not cuda_available():

            raise RuntimeError(
                "FULL_GPU requires a CUDA-capable GPU. "
                "Select LIGHT_CPU or AUTO on this machine."
            )

    return requested


def get_profile_info(
    profile: str | None = None
) -> Dict[str, Any]:

    resolved = resolve_profile(profile)

    if resolved == "full_gpu":

        return {
            "profile": "full_gpu",
            "label": "Full GPU",
            "model": FULL_MODEL_NAME,
            "imgsz": FULL_IMGSZ,
            "device": "cuda:0",
            "half_precision": True,
            "description":
                "Maximum-performance GPU inference."
        }

    return {
        "profile": "light_cpu",
        "label": "Light CPU",
        "model": LIGHT_MODEL_NAME,
        "imgsz": LIGHT_IMGSZ,
        "device": "cpu",
        "half_precision": False,
        "description":
            "CPU-friendly inference for ordinary computers."
    }


def get_available_profiles() -> List[Dict[str, Any]]:
    """
    Return profiles available on this machine.
    """

    profiles = [

        {
            "value": "auto",
            "label": "Auto (Recommended)",
            "available": True,
            "help":
                "Automatically selects GPU or CPU."
        },

        {
            "value": "light_cpu",
            "label": "Light CPU",
            "available": True,
            "help":
                "YOLOv8n at 640px for CPU-friendly inference."
        },
    ]

    if cuda_available():

        profiles.append(
            {
                "value": "full_gpu",
                "label": "Full GPU",
                "available": True,
                "help":
                    f"{FULL_MODEL_NAME} at "
                    f"{FULL_IMGSZ}px using CUDA."
            }
        )

    return profiles


# ============================================================
# MODEL LOADING
# ============================================================

def get_model(
    profile: str | None = None
) -> YOLO:

    resolved = resolve_profile(profile)

    # Reuse already-loaded model
    if resolved in _models:
        return _models[resolved]

    settings = get_profile_info(resolved)

    print(
        f"Loading Store Vision AI model: "
        f"{settings['model']} "
        f"({resolved})"
    )

    model = YOLO(
        settings["model"]
    )

    _models[resolved] = model

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
# OBJECT DETECTION
# ============================================================

def detect_objects(
    frame: np.ndarray,
    confidence: float = 0.35,
    profile: str | None = None,
):

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

    resolved = resolve_profile(profile)

    settings = get_profile_info(
        resolved
    )

    model = get_model(
        resolved
    )

    predict_kwargs = {

        "source": frame,

        "conf": float(confidence),

        "imgsz": settings["imgsz"],

        "device":
            0
            if resolved == "full_gpu"
            else "cpu",

        "verbose": False,

        "max_det": 300,
    }

    # FP16 acceleration on GPU
    if resolved == "full_gpu":

        predict_kwargs["half"] = True

    results = model.predict(
        **predict_kwargs
    )

    if not results:

        return frame.copy(), []

    result = results[0]

    annotated = result.plot()

    detections = []

    names = result.names

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            conf = float(
                box.conf[0].item()
            )

            xyxy = [
                float(value)
                for value
                in box.xyxy[0].tolist()
            ]

            detections.append(

                {
                    "class_id":
                        class_id,

                    "class_name":
                        str(
                            names[class_id]
                        ),

                    "confidence":
                        conf,

                    "bbox":
                        xyxy,
                }
            )

    return annotated, detections


# ============================================================
# HIGH-LEVEL IMAGE PROCESSING
# ============================================================

def process_image(
    image_bytes: bytes,
    confidence: float = 0.35,
    profile: str | None = None,
):

    frame = decode_image(
        image_bytes
    )

    annotated, detections = detect_objects(

        frame,

        confidence=confidence,

        profile=profile,
    )

    class_counts = {}

    for detection in detections:

        label = detection[
            "class_name"
        ]

        class_counts[label] = (
            class_counts.get(label, 0)
            + 1
        )

    settings = get_profile_info(
        profile
    )

    resolved = resolve_profile(
        profile
    )

    height, width = frame.shape[:2]

    return {

        "image":
            annotated,

        "detections":
            detections,

        "count":
            len(detections),

        "class_counts":
            class_counts,

        "profile":
            resolved,

        "profile_label":
            settings["label"],

        "model":
            settings["model"],

        "device":
            settings["device"],

        "imgsz":
            settings["imgsz"],

        "image_size":
            (width, height),
    }


# ============================================================
# PEOPLE DETECTION
# ============================================================

def detect_people(
    frame: np.ndarray,
    confidence: float = 0.35,
    profile: str | None = None,
):

    if frame is None:

        raise ValueError(
            "Image frame is empty."
        )

    if not 0.0 < confidence < 1.0:

        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    resolved = resolve_profile(
        profile
    )

    settings = get_profile_info(
        resolved
    )

    model = get_model(
        resolved
    )

    predict_kwargs = {

        "source": frame,

        "conf": float(confidence),

        "imgsz":
            settings["imgsz"],

        "device":
            0
            if resolved == "full_gpu"
            else "cpu",

        # COCO class 0 = person
        "classes": [0],

        "verbose": False,

        "max_det": 300,
    }

    if resolved == "full_gpu":

        predict_kwargs["half"] = True

    results = model.predict(
        **predict_kwargs
    )

    if not results:

        return frame.copy(), []

    result = results[0]

    annotated = result.plot()

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence_value = float(
                box.conf[0].item()
            )

            xyxy = [
                float(value)
                for value
                in box.xyxy[0].tolist()
            ]

            detections.append(

                {
                    "class_id": 0,

                    "class_name":
                        "person",

                    "confidence":
                        confidence_value,

                    "bbox":
                        xyxy,
                }
            )

    return annotated, detections


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [

    "MODEL_NAME",

    "FULL_MODEL_NAME",

    "LIGHT_MODEL_NAME",

    "cuda_available",

    "resolve_profile",

    "get_profile_info",

    "get_available_profiles",

    "get_model",

    "decode_image",

    "detect_objects",

    "detect_people",

    "process_image",
]