"""
Store Vision AI
============================================================

Complete camera pipeline:

CAMERA
  |
  +--> PERSON detection + anonymous tracking
  |
  +--> PRODUCT instance detection + tracking
  |
  +--> PERSON <-> PRODUCT association
  |
  +--> PICK / CARRY / RETURN events
  |
  +--> observed_items
  |
  +--> retail_attraction_events
  |
  +--> store_visits / visitor_events
  |
  +--> checkout reconciliation
  |
  +--> security_alerts

IMPORTANT:
This system does NOT perform facial recognition.

It uses an anonymous camera-local person_track_id.

Example:
    cam-123-person-17

A security event means:
    "suspected unbilled item"

It does NOT automatically declare a person a thief.
A store operator should review the recorded evidence.

============================================================
USAGE
============================================================

python camera_pipeline.py --camera-id CAMERA_UUID

============================================================
ENVIRONMENT VARIABLES
============================================================

Required:

SUPABASE_URL
SUPABASE_KEY

Optional:

STORE_VISION_PERSON_MODEL
STORE_VISION_PRODUCT_MODEL

STORE_VISION_PRODUCT_CONFIDENCE
STORE_VISION_PERSON_CONFIDENCE

STORE_VISION_PRODUCT_IOU
STORE_VISION_PERSON_IOU

STORE_VISION_PICK_FRAMES
STORE_VISION_RETURN_FRAMES

STORE_VISION_PERSON_ITEM_DISTANCE

STORE_VISION_FPS_LIMIT

Examples:

STORE_VISION_PERSON_MODEL=yolov8n.pt
STORE_VISION_PRODUCT_MODEL=models/store_products_seg.pt

============================================================
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import supervision as sv
from supabase import Client, create_client
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PERSON_CLASS_ID = 0


PERSON_CONFIDENCE = float(
    os.getenv(
        "STORE_VISION_PERSON_CONFIDENCE",
        "0.35",
    )
)


PRODUCT_CONFIDENCE = float(
    os.getenv(
        "STORE_VISION_PRODUCT_CONFIDENCE",
        "0.25",
    )
)


PERSON_IOU = float(
    os.getenv(
        "STORE_VISION_PERSON_IOU",
        "0.50",
    )
)


PRODUCT_IOU = float(
    os.getenv(
        "STORE_VISION_PRODUCT_IOU",
        "0.50",
    )
)


PICK_CONFIRM_FRAMES = int(
    os.getenv(
        "STORE_VISION_PICK_FRAMES",
        "5",
    )
)


RETURN_CONFIRM_FRAMES = int(
    os.getenv(
        "STORE_VISION_RETURN_FRAMES",
        "5",
    )
)


PERSON_ITEM_DISTANCE = float(
    os.getenv(
        "STORE_VISION_PERSON_ITEM_DISTANCE",
        "180",
    )
)


FPS_LIMIT = float(
    os.getenv(
        "STORE_VISION_FPS_LIMIT",
        "0",
    )
)


PRODUCT_MODEL_ENV = os.getenv(
    "STORE_VISION_PRODUCT_MODEL",
    "models/store_products_seg.pt",
)


def load_products(self):
    response = (
        self.sb.table("products")
        .select("id,name,category,sku")
        .eq("user_id", self.camera["owner_id"])
        .execute()
    )

    products = response.data or []

    self.products_by_name = {
        p["name"].strip().lower(): p
        for p in products
        if p.get("name")
    }

    return products














# ============================================================
# SUPABASE
# ============================================================

def get_supabase() -> Client:
    """
    Create Supabase client.

    This camera worker normally uses the service-role key because
    it is a backend process rather than an authenticated browser user.
    """

    url = os.environ.get(
        "SUPABASE_URL"
    )

    key = os.environ.get(
        "SUPABASE_KEY"
    )

    if not url:
        raise RuntimeError(
            "SUPABASE_URL is not configured."
        )

    if not key:
        raise RuntimeError(
            "SUPABASE_KEY is not configured."
        )

    return create_client(
        url,
        key,
    )


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# CAMERA
# ============================================================

def load_camera(
    sb: Client,
    camera_id: str,
) -> dict:

    response = (
        sb
        .table("cameras")
        .select("*")
        .eq("id", camera_id)
        .single()
        .execute()
    )

    if not response.data:
        raise ValueError(
            f"No camera found with id={camera_id}"
        )

    return response.data


def open_source(
    camera: dict,
) -> cv2.VideoCapture:

    source_type = (
        camera["source_type"]
    )

    source_uri = (
        camera["source_uri"]
    )

    if source_type == "webcam":

        try:
            source = int(
                source_uri
            )
        except (
            TypeError,
            ValueError,
        ):
            source = 0

        return cv2.VideoCapture(
            source
        )

    return cv2.VideoCapture(
        source_uri
    )


# ============================================================
# GEOMETRY
# ============================================================

def bbox_center(
    bbox: List[float],
) -> Tuple[float, float]:

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bbox_distance(
    first: List[float],
    second: List[float],
) -> float:

    ax, ay = bbox_center(
        first
    )

    bx, by = bbox_center(
        second
    )

    return (
        (
            (ax - bx) ** 2
            +
            (ay - by) ** 2
        )
        ** 0.5
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
# LINE CROSSING
# ============================================================

def line_side(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:

    px, py = point

    x1, y1 = line_start

    x2, y2 = line_end

    return (
        (x2 - x1) * (py - y1)
        -
        (y2 - y1) * (px - x1)
    )


class LineCrossingDetector:

    def __init__(
        self,
        camera: dict,
    ):

        self.start = (
            float(camera["line_x1"]),
            float(camera["line_y1"]),
        )

        self.end = (
            float(camera["line_x2"]),
            float(camera["line_y2"]),
        )

        self.previous_side = {}


    def update(
        self,
        tracker_id: int,
        bbox: List[float],
    ) -> Optional[str]:

        center = bbox_center(
            bbox
        )

        current_side = line_side(
            center,
            self.start,
            self.end,
        )

        previous = self.previous_side.get(
            tracker_id
        )

        self.previous_side[
            tracker_id
        ] = current_side

        if previous is None:
            return None

        # Ignore points directly on the line.
        if abs(current_side) < 1e-6:
            return None

        if abs(previous) < 1e-6:
            return None

        if (
            previous < 0
            and current_side > 0
        ):

            return "entry"

        if (
            previous > 0
            and current_side < 0
        ):

            return "exit"

        return None


# ============================================================
# VISIT TRACKER
# ============================================================

class VisitTracker:

    def __init__(
        self,
        sb: Client,
        camera: dict,
    ):

        self.sb = sb

        self.camera = camera

        self.open_visits: Dict[
            int,
            str,
        ] = {}


    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    def on_entry(
        self,
        tracker_id: int,
    ) -> Optional[str]:

        if tracker_id in self.open_visits:

            return self.open_visits[
                tracker_id
            ]

        person_track_id = (
            f"{self.camera['id']}-"
            f"person-{tracker_id}"
        )

        row = {

            "owner_id":
                self.camera["owner_id"],

            "store_id":
                self.camera["store_id"],

            "camera_id":
                self.camera["id"],

            "person_track_id":
                person_track_id,

            "entered_at":
                utc_now_iso(),

            "status":
                "active",

            "observed_item_count":
                0,

            "billed_item_count":
                0,

            "metadata":
                {
                    "source":
                        "camera_pipeline",
                },
        }

        result = (
            self.sb
            .table("store_visits")
            .insert(row)
            .execute()
        )

        if not result.data:

            raise RuntimeError(
                "Could not create store visit."
            )

        visit_id = result.data[0]["id"]

        self.open_visits[
            tracker_id
        ] = visit_id

        self._visitor_event(
            event_type="entry",
            person_track_id=person_track_id,
        )

        print(
            f"[ENTRY] "
            f"person={person_track_id} "
            f"visit={visit_id}"
        )

        return visit_id


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    def on_exit(
        self,
        tracker_id: int,
    ) -> Optional[str]:

        visit_id = self.open_visits.pop(
            tracker_id,
            None,
        )

        if visit_id is None:

            return None

        person_track_id = (
            f"{self.camera['id']}-"
            f"person-{tracker_id}"
        )

        self.sb.table(
            "store_visits"
        ).update({

            "status":
                "completed",

            "exited_at":
                utc_now_iso(),

        }).eq(
            "id",
            visit_id,
        ).execute()

        self._visitor_event(
            event_type="exit",
            person_track_id=person_track_id,
        )

        print(
            f"[EXIT] "
            f"person={person_track_id} "
            f"visit={visit_id}"
        )

        # ----------------------------------------------------
        # Reconcile observed items with billing.
        # ----------------------------------------------------

        try:

            self.sb.rpc(
                "fn_reconcile_visit",
                {
                    "p_visit_id":
                        visit_id,
                },
            ).execute()

            print(
                f"[RECONCILE] "
                f"visit={visit_id}"
            )

        except Exception as exc:

            print(
                "[RECONCILE ERROR]",
                exc,
            )

        return visit_id


    # --------------------------------------------------------
    # VISITOR EVENT
    # --------------------------------------------------------

    def _visitor_event(
        self,
        event_type: str,
        person_track_id: str,
    ):

        self.sb.table(
            "visitor_events"
        ).insert({

            "owner_id":
                self.camera["owner_id"],

            "store_id":
                self.camera["store_id"],

            "event_type":
                event_type,

            "camera_id":
                str(self.camera["id"]),

            "person_track_id":
                person_track_id,

            "occurred_at":
                utc_now_iso(),

            "metadata":
                {
                    "camera_name":
                        self.camera["name"],
                },

        }).execute()


# ============================================================
# PRODUCT STATE
# ============================================================

class ProductState:

    def __init__(
        self,
        tracker_id: int,
        class_name: str,
        confidence: float,
        bbox: List[float],
    ):

        self.tracker_id = tracker_id

        self.class_name = class_name

        self.confidence = confidence

        self.bbox = bbox

        self.person_track_id: Optional[
            str
        ] = None

        self.person_tracker_id: Optional[
            int
        ] = None

        self.near_person_frames = 0

        self.away_frames = 0

        self.status = "observed"

        self.observed_item_id: Optional[
            str
        ] = None

        self.first_seen_at = (
            utc_now_iso()
        )

        self.last_seen_at = (
            utc_now_iso()
        )


# ============================================================
# PRODUCT EVENT ENGINE
# ============================================================

class ProductEventEngine:

    def __init__(
        self,
        sb: Client,
        camera: dict,
    ):

        self.sb = sb

        self.camera = camera

        self.states: Dict[
            int,
            ProductState,
        ] = {}

        self.product_cache: Dict[
            str,
            Optional[str],
        ] = {}

        self.attraction_enabled = True


    # ========================================================
    # PRODUCT LOOKUP
    # ========================================================

    def find_product_id(
        self,
        label: str,
    ) -> Optional[str]:

        normalized = label.strip().lower()

        if normalized in self.product_cache:

            return self.product_cache[
                normalized
            ]

        try:

            response = (
                self.sb
                .table("products")
                .select("id,name")
                .eq(
                    "store_id",
                    self.camera["store_id"],
                )
                .execute()
            )

            product_id = None

            if response.data:

                for product in response.data:

                    name = str(
                        product.get(
                            "name",
                            "",
                        )
                    ).strip().lower()

                    if name == normalized:

                        product_id = (
                            product["id"]
                        )

                        break

            self.product_cache[
                normalized
            ] = product_id

            return product_id

        except Exception as exc:

            print(
                "[PRODUCT LOOKUP ERROR]",
                exc,
            )

            self.product_cache[
                normalized
            ] = None

            return None


    # ========================================================
    # PERSON ASSOCIATION
    # ========================================================

    def associate_person(
        self,
        product_bbox: List[float],
        persons: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        product_center = bbox_center(
            product_bbox
        )

        best_person = None

        best_distance = float(
            "inf"
        )

        for person in persons:

            person_bbox = person.get(
                "bbox"
            )

            if not person_bbox:

                continue

            person_center = bbox_center(
                person_bbox
            )

            # If the product center is inside the
            # person box, this is a strong association.
            if point_inside_bbox(
                product_center,
                person_bbox,
            ):

                return person

            current_distance = bbox_distance(
                product_bbox,
                person_bbox,
            )

            if (
                current_distance
                <
                best_distance
            ):

                best_distance = (
                    current_distance
                )

                best_person = person

        if (
            best_person is not None
            and
            best_distance
            <=
            PERSON_ITEM_DISTANCE
        ):

            return best_person

        return None


    # ========================================================
    # PROCESS PRODUCTS
    # ========================================================

    def process(
        self,
        detections: List[Dict[str, Any]],
        persons: List[Dict[str, Any]],
        active_visit_ids: Dict[int, str],
        frame=None,
    ) -> List[Dict[str, Any]]:

        events = []

        current_ids = set()

        for detection in detections:

            confidence = float(
                detection.get(
                    "confidence",
                    0,
                )
            )

            if (
                confidence
                <
                PRODUCT_CONFIDENCE
            ):

                continue

            bbox = detection.get(
                "bbox"
            )

            if not bbox:

                continue

            tracker_id = int(
                detection.get(
                    "track_id",
                    detection.get(
                        "instance_id",
                        -1,
                    ),
                )
            )

            if tracker_id < 0:

                continue

            class_name = str(
                detection.get(
                    "class_name",
                    "unknown",
                )
            )

            current_ids.add(
                tracker_id
            )

            if tracker_id not in self.states:

                self.states[
                    tracker_id
                ] = ProductState(
                    tracker_id=
                        tracker_id,
                    class_name=
                        class_name,
                    confidence=
                        confidence,
                    bbox=
                        bbox,
                )

            state = self.states[
                tracker_id
            ]

            state.bbox = bbox

            state.confidence = (
                confidence
            )

            state.last_seen_at = (
                utc_now_iso()
            )

            person = (
                self.associate_person(
                    bbox,
                    persons,
                )
            )

            if person is not None:

                person_tracker_id = (
                    person.get(
                        "track_id"
                    )
                )

                if (
                    person_tracker_id
                    is not None
                ):

                    state.person_tracker_id = (
                        int(
                            person_tracker_id
                        )
                    )

                    state.person_track_id = (
                        f"{self.camera['id']}-"
                        f"person-"
                        f"{person_tracker_id}"
                    )

                    state.near_person_frames += 1

                    state.away_frames = 0

            else:

                state.away_frames += 1

                state.near_person_frames = 0


            # =================================================
            # APPROACH
            # =================================================

            if (
                state.near_person_frames == 1
            ):

                events.append(
                    self._create_attraction_event(
                        state,
                        "approach",
                    )
                )


            # =================================================
            # PICK
            # =================================================

            if (
                state.status == "observed"
                and
                state.person_track_id
                and
                state.near_person_frames
                >=
                PICK_CONFIRM_FRAMES
            ):

                state.status = "carried"

                visit_id = None

                if (
                    state.person_tracker_id
                    is not None
                ):

                    visit_id = (
                        active_visit_ids.get(
                            state.person_tracker_id
                        )
                    )

                if visit_id:

                    observed_item = (
                        self._create_observed_item(
                            state,
                            visit_id,
                        )
                    )

                    if observed_item:

                        state.observed_item_id = (
                            observed_item["id"]
                        )

                        events.append(
                            {
                                "event_type":
                                    "pick",

                                "visit_id":
                                    visit_id,

                                "observed_item_id":
                                    observed_item["id"],

                                "person_track_id":
                                    state.person_track_id,

                                "detected_label":
                                    state.class_name,
                            }
                        )

                self._create_attraction_event(
                    state,
                    "pick",
                )


            # =================================================
            # CARRY
            # =================================================

            if (
                state.status == "carried"
                and
                state.person_track_id
            ):

                self._update_observed_item(
                    state
                )


            # =================================================
            # RETURN
            # =================================================

            if (
                state.status == "carried"
                and
                state.away_frames
                >=
                RETURN_CONFIRM_FRAMES
            ):

                state.status = "returned"

                self._mark_returned(
                    state
                )

                self._create_attraction_event(
                    state,
                    "return",
                )

                events.append(
                    {
                        "event_type":
                            "return",

                        "observed_item_id":
                            state.observed_item_id,

                        "person_track_id":
                            state.person_track_id,

                        "detected_label":
                            state.class_name,
                    }
                )


        return events


    # ========================================================
    # OBSERVED ITEM INSERT
    # ========================================================

    def _create_observed_item(
        self,
        state: ProductState,
        visit_id: str,
    ) -> Optional[Dict[str, Any]]:

        product_id = (
            self.find_product_id(
                state.class_name
            )
        )

        row = {

            "owner_id":
                self.camera["owner_id"],

            "store_id":
                self.camera["store_id"],

            "visit_id":
                visit_id,

            "camera_id":
                self.camera["id"],

            "person_track_id":
                state.person_track_id,

            "product_id":
                product_id,

            "detected_label":
                state.class_name,

            "quantity":
                1,

            "confidence":
                state.confidence,

            "picked_at":
                utc_now_iso(),

            "status":
                "observed",

            "first_seen_at":
                state.first_seen_at,

            "last_seen_at":
                state.last_seen_at,

            "bbox":
                state.bbox,

            "centroid":
                list(
                    bbox_center(
                        state.bbox
                    )
                ),

            "metadata":
                {
                    "camera_name":
                        self.camera["name"],

                    "product_tracker_id":
                        state.tracker_id,

                    "source":
                        "live_camera",
                },
        }

        try:

            response = (
                self.sb
                .table("observed_items")
                .insert(row)
                .execute()
            )

            if response.data:

                item = response.data[0]

                print(
                    "[ITEM PICKED] "
                    f"{state.class_name} "
                    f"person={state.person_track_id} "
                    f"visit={visit_id}"
                )

                return item

        except Exception as exc:

            print(
                "[OBSERVED ITEM ERROR]",
                exc,
            )

        return None


    # ========================================================
    # UPDATE OBSERVED ITEM
    # ========================================================

    def _update_observed_item(
        self,
        state: ProductState,
    ):

        if not state.observed_item_id:

            return

        try:

            self.sb.table(
                "observed_items"
            ).update({

                "last_seen_at":
                    state.last_seen_at,

                "confidence":
                    state.confidence,

                "bbox":
                    state.bbox,

                "centroid":
                    list(
                        bbox_center(
                            state.bbox
                        )
                    ),

            }).eq(
                "id",
                state.observed_item_id,
            ).execute()

        except Exception as exc:

            print(
                "[OBSERVED ITEM UPDATE ERROR]",
                exc,
            )


    # ========================================================
    # RETURN ITEM
    # ========================================================

    def _mark_returned(
        self,
        state: ProductState,
    ):

        if not state.observed_item_id:

            return

        try:

            self.sb.table(
                "observed_items"
            ).update({

                "status":
                    "returned",

                "returned_at":
                    utc_now_iso(),

                "last_seen_at":
                    state.last_seen_at,

            }).eq(
                "id",
                state.observed_item_id,
            ).execute()

        except Exception as exc:

            print(
                "[RETURN UPDATE ERROR]",
                exc,
            )


    # ========================================================
    # ATTRACTION EVENT
    # ========================================================

    def _create_attraction_event(
        self,
        state: ProductState,
        event_type: str,
    ) -> Optional[Dict[str, Any]]:

        if not self.attraction_enabled:

            return None

        row = {

            "owner_id":
                self.camera["owner_id"],

            "store_id":
                self.camera["store_id"],

            "product_id":
                self.find_product_id(
                    state.class_name
                ),

            "detected_label":
                state.class_name,

            "person_track_id":
                state.person_track_id,

            "event_type":
                event_type,

            "started_at":
                utc_now_iso(),

            "confidence":
                state.confidence,

            "metadata":
                {
                    "camera_id":
                        self.camera["id"],

                    "product_tracker_id":
                        state.tracker_id,

                    "bbox":
                        state.bbox,
                },
        }

        try:

            response = (
                self.sb
                .table(
                    "retail_attraction_events"
                )
                .insert(row)
                .execute()
            )

            if response.data:

                return response.data[0]

        except Exception as exc:

            # The camera pipeline should continue even if
            # analytics storage temporarily fails.
            print(
                "[ATTRACTION EVENT ERROR]",
                exc,
            )

        return None


# ============================================================
# MODEL HELPERS
# ============================================================

def load_model(
    path: str,
) -> YOLO:

    if not path:

        raise ValueError(
            "Model path is empty."
        )

    print(
        f"[MODEL] Loading {path}"
    )

    return YOLO(
        path
    )


# ============================================================
# PERSON DETECTION
# ============================================================

def detect_persons(
    model: YOLO,
    frame,
    confidence: float,
) -> List[Dict[str, Any]]:

    result = model.track(
        source=frame,
        persist=True,
        conf=confidence,
        iou=PERSON_IOU,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )[0]

    persons = []

    if result.boxes is None:

        return persons

    for box in result.boxes:

        if box.id is None:

            continue

        tracker_id = int(
            box.id[0].item()
        )

        bbox = [
            float(value)
            for value
            in box.xyxy[0].tolist()
        ]

        conf = float(
            box.conf[0].item()
        )

        persons.append({

            "track_id":
                tracker_id,

            "bbox":
                bbox,

            "confidence":
                conf,

        })

    return persons


# ============================================================
# PRODUCT DETECTION
# ============================================================

def detect_products(
    model: YOLO,
    frame,
    confidence: float,
) -> List[Dict[str, Any]]:

    result = model.track(
        source=frame,
        persist=True,
        conf=confidence,
        iou=PRODUCT_IOU,
        max_det=1000,
        verbose=False,
    )[0]

    detections = []

    if result.boxes is None:

        return detections

    names = result.names

    for index, box in enumerate(
        result.boxes
    ):

        bbox = [
            float(value)
            for value
            in box.xyxy[0].tolist()
        ]

        class_id = int(
            box.cls[0].item()
        )

        confidence_value = float(
            box.conf[0].item()
        )

        if box.id is not None:

            tracker_id = int(
                box.id[0].item()
            )

        else:

            tracker_id = index

        class_name = str(
            names[class_id]
        )

        detections.append({

            "track_id":
                tracker_id,

            "instance_id":
                index,

            "class_id":
                class_id,

            "class_name":
                class_name,

            "confidence":
                confidence_value,

            "bbox":
                bbox,

        })

    return detections


# ============================================================
# DRAWING
# ============================================================

def draw_persons(
    frame,
    persons,
):

    for person in persons:

        x1, y1, x2, y2 = [
            int(value)
            for value
            in person["bbox"]
        ]

        track_id = (
            person["track_id"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Person {track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def draw_products(
    frame,
    products,
):

    for product in products:

        x1, y1, x2, y2 = [
            int(value)
            for value
            in product["bbox"]
        ]

        label = (
            f"{product['class_name']} "
            f"{product['confidence']:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, min(
                frame.shape[0] - 5,
                y2 + 18,
            )),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


def draw_line(
    frame,
    camera: dict,
):

    cv2.line(
        frame,
        (
            int(camera["line_x1"]),
            int(camera["line_y1"]),
        ),
        (
            int(camera["line_x2"]),
            int(camera["line_y2"]),
        ),
        (255, 255, 255),
        3,
    )


# ============================================================
# PIPELINE
# ============================================================

def run(
    camera_id: str,
):

    # --------------------------------------------------------
    # Connect
    # --------------------------------------------------------

    sb = get_supabase()

    camera = load_camera(
        sb,
        camera_id,
    )

    if not camera["enabled"]:

        print(
            f"Camera '{camera['name']}' "
            "is disabled."
        )

        return


    # --------------------------------------------------------
    # Model paths
    # --------------------------------------------------------

    person_model_path = (
        os.getenv(
            "STORE_VISION_PERSON_MODEL"
        )
        or camera.get(
            "model_path"
        )
        or "yolov8n.pt"
    )

    product_model_path = (
        os.getenv(
            "STORE_VISION_PRODUCT_MODEL"
        )
        or PRODUCT_MODEL_ENV
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    person_model = load_model(
        person_model_path
    )

    product_model = load_model(
        product_model_path
    )


    # --------------------------------------------------------
    # Trackers
    # --------------------------------------------------------

    visits = VisitTracker(
        sb,
        camera,
    )

    line_detector = (
        LineCrossingDetector(
            camera
        )
    )

    item_engine = (
        ProductEventEngine(
            sb,
            camera,
        )
    )


    # --------------------------------------------------------
    # Camera source
    # --------------------------------------------------------

    cap = open_source(
        camera
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        int(camera["frame_width"]),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        int(camera["frame_height"]),
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera source: "
            f"{camera['source_uri']}"
        )


    print(
        "=================================================="
    )

    print(
        "Store Vision AI camera pipeline started"
    )

    print(
        f"Camera: {camera['name']}"
    )

    print(
        f"Person model: {person_model_path}"
    )

    print(
        f"Product model: {product_model_path}"
    )

    print(
        f"Product confidence: "
        f"{PRODUCT_CONFIDENCE}"
    )

    print(
        f"Person confidence: "
        f"{PERSON_CONFIDENCE}"
    )

    print(
        "=================================================="
    )


    last_frame_time = 0.0


    try:

        while True:

            # ------------------------------------------------
            # FPS LIMIT
            # ------------------------------------------------

            if FPS_LIMIT > 0:

                minimum_interval = (
                    1.0 / FPS_LIMIT
                )

                now = time.time()

                elapsed = (
                    now - last_frame_time
                )

                if (
                    elapsed
                    <
                    minimum_interval
                ):

                    time.sleep(
                        minimum_interval
                        -
                        elapsed
                    )

                last_frame_time = (
                    time.time()
                )


            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                if (
                    camera["source_type"]
                    ==
                    "file"
                ):

                    print(
                        "Video file ended."
                    )

                    break

                print(
                    "Frame read failed. "
                    "Retrying..."
                )

                time.sleep(
                    0.5
                )

                continue


            # ------------------------------------------------
            # PERSON DETECTION
            # ------------------------------------------------

            try:

                persons = detect_persons(
                    person_model,
                    frame,
                    PERSON_CONFIDENCE,
                )

            except Exception as exc:

                print(
                    "[PERSON MODEL ERROR]",
                    exc,
                )

                persons = []


            # ------------------------------------------------
            # ENTRY / EXIT
            # ------------------------------------------------

            for person in persons:

                tracker_id = (
                    person["track_id"]
                )

                crossing = (
                    line_detector.update(
                        tracker_id,
                        person["bbox"],
                    )
                )

                if crossing == "entry":

                    visits.on_entry(
                        tracker_id
                    )

                elif crossing == "exit":

                    visits.on_exit(
                        tracker_id
                    )


            # ------------------------------------------------
            # PRODUCT DETECTION
            # ------------------------------------------------

            try:

                products = detect_products(
                    product_model,
                    frame,
                    PRODUCT_CONFIDENCE,
                )

            except Exception as exc:

                print(
                    "[PRODUCT MODEL ERROR]",
                    exc,
                )

                products = []


            # ------------------------------------------------
            # PRODUCT -> PERSON EVENTS
            # ------------------------------------------------

            try:

                item_events = (
                    item_engine.process(
                        detections=products,
                        persons=persons,
                        active_visit_ids=
                            visits.open_visits,
                        frame=frame,
                    )
                )

            except Exception as exc:

                print(
                    "[ITEM ENGINE ERROR]",
                    exc,
                )

                item_events = []


            # ------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------

            draw_line(
                frame,
                camera,
            )

            draw_persons(
                frame,
                persons,
            )

            draw_products(
                frame,
                products,
            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                (
                    f"People: {len(persons)} | "
                    f"Products: {len(products)} | "
                    f"Open visits: "
                    f"{len(visits.open_visits)}"
                ),
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )


            # ------------------------------------------------
            # OPTIONAL LOCAL WINDOW
            # ------------------------------------------------

            display_enabled = (
                os.getenv(
                    "STORE_VISION_DISPLAY",
                    "0",
                )
                == "1"
            )

            if display_enabled:

                cv2.imshow(
                    "Store Vision AI",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                if key == ord("q"):

                    print(
                        "Stopped by user."
                    )

                    break


    except KeyboardInterrupt:

        print(
            "Stopped by user."
        )


    finally:

        cap.release()

        cv2.destroyAllWindows()

        print(
            "Camera pipeline stopped."
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Store Vision AI "
            "camera pipeline"
        )
    )

    parser.add_argument(
        "--camera-id",
        required=True,
        help=(
            "UUID of the camera row "
            "in Supabase"
        ),
    )

    args = parser.parse_args()

    run(
        args.camera_id
    )


if __name__ == "__main__":

    main()