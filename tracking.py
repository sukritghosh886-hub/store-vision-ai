from __future__ import annotations

import math


class CentroidTracker:

    def __init__(
        self,
        max_distance: float = 80,
    ):

        self.max_distance = max_distance

        self.next_id = 1

        self.objects = {}

    @staticmethod
    def center(bbox):

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )

    def update(self, detections):

        people = [
            detection
            for detection in detections
            if detection["label"] == "person"
        ]

        centers = [
            self.center(
                detection["bbox"]
            )
            for detection in people
        ]

        updated = {}

        used_ids = set()

        for center in centers:

            best_id = None

            best_distance = (
                self.max_distance
            )

            for object_id, old_center in (
                self.objects.items()
            ):

                if object_id in used_ids:
                    continue

                distance = math.dist(
                    center,
                    old_center,
                )

                if distance < best_distance:

                    best_distance = distance

                    best_id = object_id

            if best_id is None:

                best_id = self.next_id

                self.next_id += 1

            updated[best_id] = center

            used_ids.add(best_id)

        self.objects = updated

        return self.objects