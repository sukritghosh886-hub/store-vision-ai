"""
Store Vision AI
Person <-> Product Event Engine

Purpose:
    Connect anonymous person tracks with detected product instances.

This does NOT identify a person by face.

It creates an anonymous person_track_id such as:
    camera123-person-42

Product events:
    approach
    inspect
    pick
    carry
    return

The engine is deliberately conservative.
It produces events for downstream review instead of automatically
declaring a person a thief.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

PERSON_ITEM_DISTANCE = 180.0

PICK_CONFIRM_FRAMES = 5

RETURN_CONFIRM_FRAMES = 5

MIN_PRODUCT_CONFIDENCE = 0.35

STALE_SECONDS = 5.0


# ============================================================
# GEOMETRY
# ============================================================

def center_of_bbox(
    bbox: List[float],
) -> Tuple[float, float]:

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:

    return math.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


def point_inside_bbox(
    point: Tuple[float, float],
    bbox: List[float],
) -> bool:

    x, y = point

    x1, y1, x2, y2 = bbox

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ============================================================
# INTERNAL STATE
# ============================================================

@dataclass
class ItemState:

    tracker_id: str

    class_name: str

    confidence: float

    bbox: List[float]

    person_track_id: Optional[str] = None

    near_person_frames: int = 0

    away_frames: int = 0

    status: str = "observed"

    first_seen: float = 0.0

    last_seen: float = 0.0

    observed_item_id: Optional[str] = None


# ============================================================
# EVENT ENGINE
# ============================================================

class ItemEventEngine:

    def __init__(
        self,
        camera_id: str,
        store_id: str,
        owner_id: str,
    ):

        self.camera_id = camera_id

        self.store_id = store_id

        self.owner_id = owner_id

        self.items: Dict[str, ItemState] = {}


    # ========================================================
    # PERSON ASSOCIATION
    # ========================================================

    def find_nearest_person(
        self,
        item_bbox: List[float],
        persons: List[Dict[str, Any]],
    ) -> Optional[str]:

        item_center = center_of_bbox(
            item_bbox
        )

        best_person = None

        best_distance = float("inf")

        for person in persons:

            person_bbox = person.get(
                "bbox"
            )

            person_track_id = person.get(
                "track_id"
            )

            if (
                not person_bbox
                or person_track_id is None
            ):
                continue

            person_center = center_of_bbox(
                person_bbox
            )

            d = distance(
                item_center,
                person_center,
            )

            # Stronger association when item is
            # physically inside the person box.
            if point_inside_bbox(
                item_center,
                person_bbox,
            ):

                return str(
                    person_track_id
                )

            if d < best_distance:

                best_distance = d

                best_person = (
                    str(person_track_id)
                )

        if (
            best_person is not None
            and
            best_distance <= PERSON_ITEM_DISTANCE
        ):

            return best_person

        return None


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process(
        self,
        detections: List[Dict[str, Any]],
        persons: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        events = []

        seen_ids = set()

        for detection in detections:

            confidence = float(
                detection.get(
                    "confidence",
                    0.0
                )
            )

            if (
                confidence
                <
                MIN_PRODUCT_CONFIDENCE
            ):
                continue

            bbox = detection.get(
                "bbox"
            )

            if not bbox:
                continue

            class_name = str(
                detection.get(
                    "class_name",
                    "unknown"
                )
            )

            # Use model tracker ID if available.
            tracker_id = str(
                detection.get(
                    "track_id",
                    detection.get(
                        "instance_id"
                    )
                )
            )

            seen_ids.add(
                tracker_id
            )

            person_id = (
                self.find_nearest_person(
                    bbox,
                    persons
                )
            )

            if tracker_id not in self.items:

                self.items[tracker_id] = ItemState(
                    tracker_id=tracker_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    person_track_id=person_id,
                    first_seen=now,
                    last_seen=now,
                )

                if person_id:

                    events.append(
                        self._event(
                            "approach",
                            self.items[
                                tracker_id
                            ],
                            now,
                        )
                    )

                continue

            state = self.items[
                tracker_id
            ]

            state.bbox = bbox

            state.confidence = confidence

            state.last_seen = now

            if person_id:

                state.person_track_id = (
                    person_id
                )

                state.near_person_frames += 1

                state.away_frames = 0

            else:

                state.away_frames += 1

                state.near_person_frames = 0

            # ------------------------------------------------
            # PICK CONFIRMATION
            # ------------------------------------------------

            if (
                state.status == "observed"
                and
                state.person_track_id
                and
                state.near_person_frames
                >= PICK_CONFIRM_FRAMES
            ):

                state.status = "carried"

                events.append(
                    self._event(
                        "pick",
                        state,
                        now,
                    )
                )

                events.append(
                    self._event(
                        "carry",
                        state,
                        now,
                    )
                )

            # ------------------------------------------------
            # RETURN
            # ------------------------------------------------

            elif (
                state.status == "carried"
                and
                state.away_frames
                >= RETURN_CONFIRM_FRAMES
            ):

                state.status = "returned"

                events.append(
                    self._event(
                        "return",
                        state,
                        now,
                    )
                )

        self._remove_stale(
            now
        )

        return events


    # ========================================================
    # EVENT FORMAT
    # ========================================================

    def _event(
        self,
        event_type: str,
        state: ItemState,
        timestamp: float,
    ) -> Dict[str, Any]:

        return {

            "owner_id":
                self.owner_id,

            "store_id":
                self.store_id,

            "camera_id":
                self.camera_id,

            "event_type":
                event_type,

            "person_track_id":
                state.person_track_id,

            "item_tracker_id":
                state.tracker_id,

            "detected_label":
                state.class_name,

            "confidence":
                state.confidence,

            "bbox":
                state.bbox,

            "timestamp":
                timestamp,

            "status":
                state.status,
        }


    # ========================================================
    # STALE CLEANUP
    # ========================================================

    def _remove_stale(
        self,
        now: float,
    ):

        remove_ids = []

        for tracker_id, state in self.items.items():

            if (
                now - state.last_seen
                >
                STALE_SECONDS
            ):

                remove_ids.append(
                    tracker_id
                )

        for tracker_id in remove_ids:

            del self.items[
                tracker_id
            ]


__all__ = [
    "ItemEventEngine",
    "center_of_bbox",
    "distance",
]