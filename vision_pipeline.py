import cv2
import numpy as np

from ultralytics import YOLO

import store_events
from models.person_tracker import (
    CentroidPersonTracker,
    PersonDetection,
)


PERSON_CLASS = 0

ITEM_CLASSES = {
    24: "backpack",
    26: "handbag",
    39: "bottle",
    41: "cup",
    43: "knife",
    45: "bowl",
    46: "banana",
    47: "apple",
    49: "orange",
    67: "cell_phone",
}


def _bbox_from_xyxy(box):
    x1, y1, x2, y2 = map(int, box)

    return (
        x1,
        y1,
        max(1, x2 - x1),
        max(1, y2 - y1),
    )


def _center(bbox):
    x, y, w, h = bbox

    return (
        x + w // 2,
        y + h // 2,
    )


def _distance(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)

    return float(
        np.hypot(
            ax - bx,
            ay - by,
        )
    )


def process_video(
    video_path,
    store_id,
    shelf_zone_frac=0.55,
    exit_zone_frac=0.85,
    conf=0.4,
    frame_stride=2,
):
    """
    Process an uploaded store video.

    Pipeline:

    video
       ↓
    YOLO detection
       ↓
    person tracking
       ↓
    visit creation
       ↓
    item association
       ↓
    exit detection
       ↓
    billing comparison
       ↓
    alert creation
    """

    model = YOLO("yolov8n.pt")

    tracker = CentroidPersonTracker(
        max_distance=120,
        max_missed_frames=20,
    )

    visit_ids = {}

    seen_items = {}

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the uploaded video."
        )

    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            if (
                frame_number % frame_stride
                != 0
            ):
                continue

            height, width = frame.shape[:2]

            shelf_x = int(
                width * shelf_zone_frac
            )

            exit_x = int(
                width * exit_zone_frac
            )

            results = model.predict(
                frame,
                conf=conf,
                verbose=False,
            )

            person_detections = []

            item_detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    bbox = _bbox_from_xyxy(
                        box.xyxy[0].tolist()
                    )

                    if cls == PERSON_CLASS:
                        person_detections.append(
                            PersonDetection(
                                bbox=bbox,
                                confidence=confidence,
                            )
                        )

                    elif cls in ITEM_CLASSES:
                        item_detections.append(
                            (
                                ITEM_CLASSES[cls],
                                bbox,
                                confidence,
                            )
                        )

            tracks = tracker.update(
                person_detections
            )

            active_track_ids = set()

            for track in tracks:
                active_track_ids.add(
                    track.track_id
                )

                if track.track_id not in visit_ids:
                    visit_ids[
                        track.track_id
                    ] = store_events.start_visit(
                        store_id=store_id,
                        track_id=track.track_id,
                    )

                    seen_items[
                        track.track_id
                    ] = set()

                visit_id = visit_ids[
                    track.track_id
                ]

                px, py = track.center

                # Associate detected objects with
                # the nearest tracked person.
                for (
                    item_label,
                    item_bbox,
                    item_confidence,
                ) in item_detections:

                    distance = _distance(
                        track.bbox,
                        item_bbox,
                    )

                    if distance > 180:
                        continue

                    ix, iy = _center(
                        item_bbox
                    )

                    # Items detected in the shelf zone
                    # are considered observed/picked items.
                    if ix >= shelf_x:
                        key = (
                            item_label,
                            track.track_id,
                        )

                        if key not in seen_items[
                            track.track_id
                        ]:
                            store_events.log_item_event(
                                visit_id=visit_id,
                                item_label=item_label,
                                zone="shelf",
                                confidence=item_confidence,
                            )

                            seen_items[
                                track.track_id
                            ].add(key)

                # Draw person.
                x, y, w, h = track.bbox

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Person #{track.track_id}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Exit handling.
                if px >= exit_x:
                    alert = store_events.close_visit(
                        visit_id,
                        store_id,
                    )

                    if alert:
                        yield (
                            frame,
                            (
                                f"Track #{track.track_id} "
                                f"flagged for review: "
                                f"{alert['unpaid_item_count']} "
                                f"unpaid item(s)."
                            ),
                        )
                    else:
                        yield (
                            frame,
                            (
                                f"Track #{track.track_id} "
                                f"exited cleanly."
                            ),
                        )

                    del visit_ids[
                        track.track_id
                    ]

                    seen_items.pop(
                        track.track_id,
                        None,
                    )

            # Draw zones.
            cv2.line(
                frame,
                (shelf_x, 0),
                (shelf_x, height),
                (255, 200, 0),
                2,
            )

            cv2.line(
                frame,
                (exit_x, 0),
                (exit_x, height),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                "SHELF",
                (shelf_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
            )

            cv2.putText(
                frame,
                "EXIT",
                (exit_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            yield frame, None

    finally:
        cap.release()