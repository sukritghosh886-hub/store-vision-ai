from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import time


BBox = Tuple[float, float, float, float]


@dataclass
class CarryObservation:
    track_id: int
    item_name: str
    frames: int = 0
    last_seen: float = 0.0
    near_exit: bool = False


@dataclass
class TheftAlert:
    track_id: int
    item_name: str
    reason: str
    confidence: float
    created_at: float


class TheftDetector:
    """
    Detects SUSPECTED THEFT using:

        person tracking
        +
        product tracking
        +
        exit-zone detection
        +
        payment information

    This does NOT perform facial recognition.
    """

    def __init__(
        self,
        exit_zone: BBox,
        association_distance: float = 180.0,
        min_carry_frames: int = 8,
        stale_seconds: float = 5.0,
    ):

        self.exit_zone = exit_zone

        self.association_distance = association_distance

        self.min_carry_frames = min_carry_frames

        self.stale_seconds = stale_seconds

        # (person_id, product_name)
        self.carry: Dict[
            Tuple[int, str],
            CarryObservation
        ] = {}

        # Prevent the same event from generating
        # hundreds of alerts.
        self.alerted = set()

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    @staticmethod
    def center(
        bbox: BBox
    ) -> Tuple[float, float]:

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    @staticmethod
    def distance(
        point_a,
        point_b
    ) -> float:

        return math.hypot(
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
        )

    # --------------------------------------------------
    # Exit zone
    # --------------------------------------------------

    def inside_exit_zone(
        self,
        bbox: BBox
    ) -> bool:

        cx, cy = self.center(bbox)

        x1, y1, x2, y2 = self.exit_zone

        return (
            x1 <= cx <= x2
            and
            y1 <= cy <= y2
        )

    # --------------------------------------------------
    # Associate product with nearest person
    # --------------------------------------------------

    def associate_products(
        self,
        people: List[dict],
        products: List[dict],
    ):

        associations = []

        for product in products:

            product_center = self.center(
                tuple(product["bbox"])
            )

            nearest_person = None

            nearest_distance = float("inf")

            for person in people:

                person_center = self.center(
                    tuple(person["bbox"])
                )

                current_distance = self.distance(
                    product_center,
                    person_center,
                )

                if current_distance < nearest_distance:

                    nearest_distance = current_distance

                    nearest_person = person

            if (
                nearest_person is not None
                and
                nearest_distance
                <= self.association_distance
            ):

                associations.append(
                    (
                        nearest_person,
                        product,
                        nearest_distance,
                    )
                )

        return associations

    # --------------------------------------------------
    # Main detector
    # --------------------------------------------------

    def update(
        self,
        people: List[dict],
        products: List[dict],
        paid_items: Optional[
            Dict[int, List[str]]
        ] = None,
    ) -> List[TheftAlert]:

        if paid_items is None:
            paid_items = {}

        now = time.time()

        alerts = []

        people_by_id = {
            int(person["track_id"]): person
            for person in people
        }

        current_keys = set()

        associations = self.associate_products(
            people,
            products,
        )

        for person, product, distance in associations:

            track_id = int(
                person["track_id"]
            )

            item_name = str(
                product["class_name"]
            )

            key = (
                track_id,
                item_name,
            )

            current_keys.add(key)

            observation = self.carry.get(key)

            if observation is None:

                observation = CarryObservation(
                    track_id=track_id,
                    item_name=item_name,
                    frames=0,
                    last_seen=now,
                )

                self.carry[key] = observation

            observation.frames += 1

            observation.last_seen = now

            observation.near_exit = (
                self.inside_exit_zone(
                    tuple(person["bbox"])
                )
            )

            # Person must carry the item for
            # several frames.
            if (
                observation.frames
                < self.min_carry_frames
            ):
                continue

            # Person isn't at the exit yet.
            if not observation.near_exit:
                continue

            # ------------------------------------------
            # PAYMENT CHECK
            # ------------------------------------------

            paid_for_person = paid_items.get(
                track_id,
                []
            )

            if item_name in paid_for_person:

                # Item was paid.
                continue

            # ------------------------------------------
            # DUPLICATE ALERT PROTECTION
            # ------------------------------------------

            if key in self.alerted:

                continue

            self.alerted.add(key)

            detector_confidence = float(
                product.get(
                    "confidence",
                    0.5
                )
            )

            persistence_score = min(
                1.0,
                observation.frames /
                max(
                    self.min_carry_frames * 2,
                    1
                )
            )

            confidence = min(
                0.99,
                (
                    detector_confidence
                    +
                    persistence_score
                ) / 2
            )

            alerts.append(
                TheftAlert(
                    track_id=track_id,

                    item_name=item_name,

                    reason=(
                        "The tracked item reached "
                        "the exit without a matching "
                        "payment record."
                    ),

                    confidence=confidence,

                    created_at=now,
                )
            )

        # ------------------------------------------
        # Remove stale observations
        # ------------------------------------------

        stale_keys = []

        for key, observation in self.carry.items():

            if key in current_keys:
                continue

            if (
                now -
                observation.last_seen
                >
                self.stale_seconds
            ):

                stale_keys.append(key)

        for key in stale_keys:

            del self.carry[key]

        return alerts