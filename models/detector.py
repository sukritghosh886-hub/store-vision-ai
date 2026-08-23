from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """
    Represents one detected visual object.
    """

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class ShelfDetector:
    """
    Baseline shelf detector.

    This version uses OpenCV image processing.
    A trained YOLO model can replace this class later
    without changing the rest of the application.
    """

    def __init__(self, min_area: int = 800):
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect visually separated objects using contours.

        Parameters
        ----------
        image:
            OpenCV image in BGR format.

        Returns
        -------
        List[Detection]
        """

        if image is None:
            return []

        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Close small gaps
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            processed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections = []

        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            box_area = width * height

            if box_area <= 0:
                continue

            # Basic confidence estimate.
            # This is NOT ML confidence.
            confidence = min(
                0.99,
                max(0.35, area / image_area * 8),
            )

            detections.append(
                Detection(
                    label="product",
                    confidence=float(confidence),
                    bbox=(x, y, width, height),
                )
            )

        # Largest objects first
        detections.sort(
            key=lambda item: item.bbox[2] * item.bbox[3],
            reverse=True,
        )

        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes around detections.
        """

        output = image.copy()

        for index, detection in enumerate(detections, start=1):
            x, y, width, height = detection.bbox

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = (
                f"{index}: {detection.label} "
                f"{detection.confidence:.0%}"
            )

            cv2.putText(
                output,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output