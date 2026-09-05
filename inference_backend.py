"""
Hardware-adaptive inference for Store Vision AI.

Supports:

    auto -> GPU if available, otherwise CPU
    cpu  -> lightweight CPU inference
    gpu  -> GPU inference, with CPU fallback
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ultralytics import YOLO


DEFAULT_MODEL = os.getenv(
    "STORE_VISION_MODEL",
    "yolov8n.pt",
)


class InferenceBackend:

    def __init__(
        self,
        mode: str = "auto",
        model_path: Optional[str] = None,
    ) -> None:

        self.mode = self._resolve_mode(
            mode
        )

        self.model_path = (
            model_path
            or DEFAULT_MODEL
        )

        self.model = YOLO(
            self.model_path
        )

    @staticmethod
    def gpu_available() -> bool:

        try:
            import torch

            return bool(
                torch.cuda.is_available()
            )

        except Exception:
            return False

    @classmethod
    def _resolve_mode(
        cls,
        mode: str,
    ) -> str:

        requested = (
            mode
            or "auto"
        ).lower().strip()

        if requested not in {
            "auto",
            "cpu",
            "gpu",
        }:
            raise ValueError(
                "mode must be "
                "auto, cpu, or gpu"
            )

        if requested == "auto":

            if cls.gpu_available():
                return "gpu"

            return "cpu"

        if (
            requested == "gpu"
            and not cls.gpu_available()
        ):
            return "cpu"

        return requested

    @property
    def device(self) -> str:

        if self.mode == "gpu":
            return "0"

        return "cpu"

    @property
    def image_size(self) -> int:

        if self.mode == "cpu":
            return 416

        return 640

    @property
    def default_frame_stride(self) -> int:

        if self.mode == "cpu":
            return 4

        return 2

    def predict(
        self,
        frame: Any,
        confidence: float = 0.35,
    ):

        return self.model.predict(
            source=frame,
            conf=confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

    def info(self) -> Dict[str, Any]:

        return {
            "mode": self.mode,
            "device": self.device,
            "model": self.model_path,
            "image_size": self.image_size,
            "default_frame_stride":
                self.default_frame_stride,
            "gpu_available":
                self.gpu_available(),
        }


_backend: Optional[
    InferenceBackend
] = None

_backend_key: Optional[
    Tuple[str, str]
] = None


def get_backend(
    mode: str = "auto",
    model_path: Optional[str] = None,
) -> InferenceBackend:

    global _backend
    global _backend_key

    key = (
        (mode or "auto").lower().strip(),
        model_path or DEFAULT_MODEL,
    )

    if (
        _backend is None
        or _backend_key != key
    ):

        _backend = InferenceBackend(
            mode=mode,
            model_path=model_path,
        )

        _backend_key = key

    return _backend