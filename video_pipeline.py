from __future__ import annotations

import cv2

from vision_pipeline import detect_objects
from tracking import CentroidTracker


def process_video(
    video_path: str,
    confidence: float = 0.35,
    frame_skip: int = 3,
):

    capture = cv2.VideoCapture(
        video_path
    )

    tracker = CentroidTracker()

    frame_number = 0

    try:

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            if frame_number % frame_skip != 0:
                continue

            annotated, detections = (
                detect_objects(
                    frame,
                    confidence,
                )
            )

            tracked_people = tracker.update(
                detections
            )

            yield {
                "frame_number": frame_number,
                "frame": annotated,
                "detections": detections,
                "tracked_people":
                    tracked_people,
                "people_count":
                    len(tracked_people),
            }

    finally:

        capture.release()