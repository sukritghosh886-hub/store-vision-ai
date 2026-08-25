from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from models.person_tracker import PersonDetection


@dataclass
class PersonDetectorConfig:
    """
    Configuration for the person detector.

    model_path:
        Path/name of the YOLO model.

    confidence:
        Minimum confidence required to accept a detection.

    device:
        "cpu" or a supported accelerator.
    """

    model_path: str = "yolo11n.pt"
    confidence: float = 0.40
    device: str = "cpu"


class PersonDetector:
    """
    Real person detector for Store Vision AI.

    The detector is deliberately kept separate from the tracker.

    Flow:

        camera frame
             ↓
        PersonDetector
             ↓
        PersonDetection objects
             ↓
        CentroidPersonTracker
    """

    # COCO class ID for "person".
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        config: PersonDetectorConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else PersonDetectorConfig()
        )

        self.model = None
        self._load_error = None

        self._load_model()

    def _load_model(self) -> None:
        """
        Load the YOLO model lazily.

        Keeping the error instead of crashing during import
        makes the Streamlit application easier to diagnose.
        """

        try:
            from ultralytics import YOLO

            self.model = YOLO(
                self.config.model_path
            )

        except Exception as exc:
            self._load_error = str(exc)
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect(
        self,
        frame: np.ndarray,
    ) -> List[PersonDetection]:
        """
        Detect people in one BGR OpenCV frame.

        Returns:
            List[PersonDetection]
        """

        if frame is None:
            return []

        if self.model is None:
            raise RuntimeError(
                "Person detector is not available. "
                f"Model loading error: {self._load_error}"
            )

        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            classes=[self.PERSON_CLASS_ID],
            device=self.config.device,
            verbose=False,
        )

        detections: List[PersonDetection] = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        frame_height, frame_width = frame.shape[:2]

        for box, confidence in zip(
            xyxy,
            confidences,
        ):
            x1, y1, x2, y2 = box

            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width))
            y2 = max(0, min(int(y2), frame_height))

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            detections.append(
                PersonDetection(
                    bbox=(
                        x1,
                        y1,
                        width,
                        height,
                    ),
                    confidence=float(confidence),
                )
            )

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[PersonDetection],
    ) -> np.ndarray:
        """
        Draw detected people on a frame.
        """

        output = frame.copy()

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            x, y, width, height = detection.bbox

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = (
                f"Person {index} "
                f"{detection.confidence:.0%}"
            )

            cv2.putText(
                output,
                label,
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output