from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2

import store_events

from vision_pipeline import detect_objects
from tracking import CentroidTracker


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIDENCE = 0.40

DEFAULT_FRAME_STRIDE = 2

DEFAULT_ITEM_PERSON_DISTANCE = 300.0

PRODUCT_CLASSES = {
    "book",
    "bottle",
    "cap",
    "sticker",
}


# ============================================================
# ITEM TRACKER
# ============================================================

class ItemTracker:
    """
    Lightweight anonymous tracker for product instances.

    This allows multiple copies of the same product
    to receive different temporary instance IDs.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missing: int = 10,
    ):

        self.max_distance = float(
            max_distance
        )

        self.max_missing = int(
            max_missing
        )

        self.next_id = 1

        self.objects: Dict[
            int,
            Dict
        ] = {}

    @staticmethod
    def center(
        bbox: List[float],
    ) -> Tuple[float, float]:

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    def update(
        self,
        detections: List[dict],
    ) -> Dict[int, Dict]:

        products = []

        for detection in detections:

            class_name = str(
                detection.get(
                    "class_name",
                    detection.get(
                        "label",
                        "",
                    ),
                )
            ).lower()

            if class_name in PRODUCT_CLASSES:

                products.append(
                    detection
                )

        new_items = []

        for detection in products:

            new_items.append(
                {
                    "class_name": str(
                        detection.get(
                            "class_name",
                            detection.get(
                                "label",
                                "unknown",
                            ),
                        )
                    ).lower(),

                    "center": self.center(
                        detection["bbox"]
                    ),

                    "detection": detection,
                }
            )

        existing = list(
            self.objects.items()
        )

        candidates = []

        for new_index, item in enumerate(
            new_items
        ):

            for object_id, old_item in existing:

                if (
                    item["class_name"]
                    != old_item["class_name"]
                ):
                    continue

                distance = math.dist(
                    item["center"],
                    old_item["center"],
                )

                if (
                    distance
                    <= self.max_distance
                ):

                    candidates.append(
                        (
                            distance,
                            new_index,
                            object_id,
                        )
                    )

        candidates.sort(
            key=lambda value: value[0]
        )

        assigned_new = set()
        assigned_ids = set()

        updated = {}

        for (
            _,
            new_index,
            object_id,
        ) in candidates:

            if new_index in assigned_new:
                continue

            if object_id in assigned_ids:
                continue

            item = new_items[
                new_index
            ]

            updated[object_id] = {
                "class_name":
                    item["class_name"],

                "center":
                    item["center"],

                "detection":
                    item["detection"],

                "missing": 0,
            }

            assigned_new.add(
                new_index
            )

            assigned_ids.add(
                object_id
            )

        # Create new product instances.
        for new_index, item in enumerate(
            new_items
        ):

            if new_index in assigned_new:
                continue

            object_id = self.next_id

            self.next_id += 1

            updated[object_id] = {
                "class_name":
                    item["class_name"],

                "center":
                    item["center"],

                "detection":
                    item["detection"],

                "missing": 0,
            }

        # Keep temporarily missing objects.
        for object_id, old_item in existing:

            if object_id in updated:
                continue

            missing = (
                old_item.get(
                    "missing",
                    0,
                )
                + 1
            )

            if missing <= self.max_missing:

                old_item = dict(
                    old_item
                )

                old_item["missing"] = (
                    missing
                )

                updated[object_id] = (
                    old_item
                )

        self.objects = updated

        return dict(
            self.objects
        )


# ============================================================
# HELPERS
# ============================================================

def detection_name(
    detection: dict,
) -> str:

    return str(
        detection.get(
            "class_name",
            detection.get(
                "label",
                "unknown",
            ),
        )
    ).lower()


def detection_center(
    detection: dict,
) -> Tuple[float, float]:

    if "centroid" in detection:

        centroid = detection[
            "centroid"
        ]

        return (
            float(centroid[0]),
            float(centroid[1]),
        )

    bbox = detection["bbox"]

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def nearest_person(
    item: dict,
    people: Dict[
        int,
        Tuple[float, float]
    ],
    maximum_distance: float,
):

    if not people:
        return None

    item_center = detection_center(
        item
    )

    best_id = None

    best_distance = (
        maximum_distance
    )

    for track_id, center in people.items():

        distance = math.dist(
            item_center,
            center,
        )

        if distance < best_distance:

            best_distance = distance

            best_id = track_id

    return best_id


def calculate_zone(
    detection: dict,
    frame_height: int,
    shelf_zone_frac: float,
) -> str:

    _, y = detection_center(
        detection
    )

    shelf_line = (
        frame_height
        * shelf_zone_frac
    )

    if y >= shelf_line:
        return "shelf"

    return "store"


# ============================================================
# VIDEO PIPELINE
# ============================================================

def process_video(
    video_path: str,
    store_id: str,
    shelf_zone_frac: float = 0.55,
    exit_zone_frac: float = 0.85,
    conf: float = DEFAULT_CONFIDENCE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    mode: str = "auto",
    item_person_distance: float = (
        DEFAULT_ITEM_PERSON_DISTANCE
    ),
):
    """
    Complete Store Vision AI video pipeline.

    Pipeline:

        Video
          ↓
        YOLO detection
          ↓
        Anonymous person tracking
          ↓
        Product-instance tracking
          ↓
        Person ↔ product association
          ↓
        Supabase visit/item events
          ↓
        Exit detection
          ↓
        Unpaid-item security alert
    """

    if not store_id:
        raise ValueError(
            "store_id is required."
        )

    if frame_stride < 1:
        frame_stride = 1

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    person_tracker = (
        CentroidTracker(
            max_distance=120,
            max_missing=15,
        )
    )

    item_tracker = (
        ItemTracker(
            max_distance=100,
            max_missing=10,
        )
    )

    # anonymous_track_id -> Supabase visit ID
    active_visits: Dict[
        int,
        str
    ] = {}

    # Product instance IDs already written
    logged_items = set()

    # Prevent repeatedly closing the same visit.
    closed_tracks = set()

    frame_number = 0

    inference_count = 0

    total_detections = 0

    total_people = set()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            if (
                (frame_number - 1)
                % frame_stride
                != 0
            ):
                continue

            height, width = (
                frame.shape[:2]
            )

            annotated, detections = (
                detect_objects(
                    frame,
                    confidence=conf,
                )
            )

            inference_count += 1

            total_detections += len(
                detections
            )

            # ------------------------------------------------
            # PERSON TRACKING
            # ------------------------------------------------

            people = (
                person_tracker.update(
                    detections
                )
            )

            total_people.update(
                people.keys()
            )

            # Create a visit for every newly observed person.
            for track_id in people:

                if track_id in active_visits:
                    continue

                if track_id in closed_tracks:
                    continue

                try:

                    visit_id = (
                        store_events.start_visit(
                            store_id=store_id,
                            track_id=track_id,
                            camera_id="default",
                        )
                    )

                    active_visits[
                        track_id
                    ] = visit_id

                except Exception as exc:

                    yield {
                        "frame_number":
                            frame_number,

                        "frame":
                            annotated,

                        "detections":
                            detections,

                        "tracked_people":
                            people,

                        "people_count":
                            len(people),

                        "message":
                            "Could not create visit "
                            f"for anonymous person "
                            f"{track_id}: {exc}",
                    }

                    continue

            # ------------------------------------------------
            # PRODUCT TRACKING
            # ------------------------------------------------

            tracked_items = (
                item_tracker.update(
                    detections
                )
            )

            # ------------------------------------------------
            # ITEM → PERSON ASSOCIATION
            # ------------------------------------------------

            for item_id, item in (
                tracked_items.items()
            ):

                detection = item[
                    "detection"
                ]

                person_id = nearest_person(
                    detection,
                    people,
                    item_person_distance,
                )

                if person_id is None:
                    continue

                visit_id = active_visits.get(
                    person_id
                )

                if not visit_id:
                    continue

                event_key = (
                    visit_id,
                    item_id,
                )

                # Don't insert the same tracked
                # product instance on every frame.
                if event_key in logged_items:
                    continue

                label = detection_name(
                    detection
                )

                zone = calculate_zone(
                    detection,
                    height,
                    shelf_zone_frac,
                )

                confidence = detection.get(
                    "confidence"
                )

                try:

                    store_events.log_item_event(
                        visit_id=visit_id,
                        item_label=label,
                        zone=zone,
                        confidence=confidence,
                    )

                    logged_items.add(
                        event_key
                    )

                except Exception as exc:

                    yield {
                        "frame_number":
                            frame_number,

                        "frame":
                            annotated,

                        "detections":
                            detections,

                        "tracked_people":
                            people,

                        "people_count":
                            len(people),

                        "message":
                            f"Item event error: {exc}",
                    }

            # ------------------------------------------------
            # EXIT DETECTION
            # ------------------------------------------------

            exited_tracks = []

            exit_line = (
                height
                * exit_zone_frac
            )

            for track_id, center in (
                people.items()
            ):

                _, y = center

                if y < exit_line:
                    continue

                if track_id in closed_tracks:
                    continue

                visit_id = active_visits.get(
                    track_id
                )

                if not visit_id:
                    continue

                try:

                    alert = (
                        store_events.close_visit(
                            visit_id=visit_id,
                            store_id=store_id,
                        )
                    )

                    exited_tracks.append(
                        {
                            "track_id":
                                track_id,

                            "visit_id":
                                visit_id,

                            "alert":
                                alert,
                        }
                    )

                except Exception as exc:

                    exited_tracks.append(
                        {
                            "track_id":
                                track_id,

                            "visit_id":
                                visit_id,

                            "error":
                                str(exc),
                        }
                    )

                closed_tracks.add(
                    track_id
                )

                active_visits.pop(
                    track_id,
                    None
                )

            # ------------------------------------------------
            # VISUAL STATUS
            # ------------------------------------------------

            message = (
                f"frame={frame_number} "
                f"detections={len(detections)} "
                f"people={len(people)} "
                f"active_visits={len(active_visits)} "
                f"tracked_items={len(tracked_items)} "
                f"exits={len(exited_tracks)}"
            )

            yield {
                "frame_number":
                    frame_number,

                "frame":
                    annotated,

                "detections":
                    detections,

                "tracked_people":
                    people,

                "tracked_items":
                    tracked_items,

                "people_count":
                    len(people),

                "active_visits":
                    dict(active_visits),

                "exits":
                    exited_tracks,

                "inference_count":
                    inference_count,

                "message":
                    message,
            }

    finally:

        # ----------------------------------------------------
        # CLOSE VIDEO
        # ----------------------------------------------------

        cap.release()

        # Any person still inside at the end of the video
        # is intentionally NOT automatically classified as
        # a thief. Their visit remains open.