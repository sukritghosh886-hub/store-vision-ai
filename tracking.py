from __future__ import annotations

import math
from typing import Dict, List, Tuple


class CentroidTracker:
    """
    CPU-friendly anonymous person tracker.

    This does NOT perform facial recognition.
    Each person receives a temporary anonymous ID.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_missing: int = 20,
    ):
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)

        self.next_id = 1

        self.objects: Dict[
            int,
            Tuple[float, float]
        ] = {}

        self.missing: Dict[int, int] = {}

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
    ) -> Dict[int, Tuple[float, float]]:

        people = [
            detection
            for detection in detections
            if str(
                detection.get(
                    "class_name",
                    detection.get("label", ""),
                )
            ).lower() == "person"
        ]

        centers = [
            self.center(
                detection["bbox"]
            )
            for detection in people
        ]

        # No people detected in this frame.
        if not centers:

            for object_id in list(
                self.missing
            ):
                self.missing[object_id] += 1

                if (
                    self.missing[object_id]
                    > self.max_missing
                ):
                    self.objects.pop(
                        object_id,
                        None,
                    )

                    self.missing.pop(
                        object_id,
                        None,
                    )

            return dict(self.objects)

        old_items = list(
            self.objects.items()
        )

        candidates = []

        for new_index, center in enumerate(
            centers
        ):

            for object_id, old_center in (
                old_items
            ):

                distance = math.dist(
                    center,
                    old_center,
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

        # Closest matches first.
        candidates.sort(
            key=lambda item: item[0]
        )

        assigned_new = set()
        assigned_ids = set()

        updated: Dict[
            int,
            Tuple[float, float]
        ] = {}

        for (
            _,
            new_index,
            object_id,
        ) in candidates:

            if new_index in assigned_new:
                continue

            if object_id in assigned_ids:
                continue

            updated[object_id] = (
                centers[new_index]
            )

            assigned_new.add(
                new_index
            )

            assigned_ids.add(
                object_id
            )

        # Create anonymous IDs for new people.
        for new_index, center in enumerate(
            centers
        ):

            if new_index in assigned_new:
                continue

            object_id = self.next_id

            self.next_id += 1

            updated[object_id] = center

        # Update missing counters.
        for object_id in list(
            self.objects
        ):

            if object_id not in updated:

                self.missing[object_id] = (
                    self.missing.get(
                        object_id,
                        0,
                    ) + 1
                )

            else:

                self.missing[object_id] = 0

        # Remove stale tracks.
        for object_id in list(
            self.missing
        ):

            if (
                self.missing[object_id]
                > self.max_missing
            ):

                self.missing.pop(
                    object_id,
                    None,
                )

                self.objects.pop(
                    object_id,
                    None,
                )

                updated.pop(
                    object_id,
                    None,
                )

        self.objects = updated

        return dict(self.objects)