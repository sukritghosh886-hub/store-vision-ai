"""
Store Vision AI - Camera Pipeline

Pipeline:

Camera
  -> Person detection/tracking
  -> Anonymous person ID
  -> Product detection/tracking
  -> Person/product association
  -> observed_items
  -> retail_attraction_events
  -> Visit exit
  -> fn_reconcile_visit()
  -> billing comparison
  -> security_alerts

IMPORTANT:
This system creates SUSPECTED UNBILLED ITEM alerts.
It does not automatically identify or label a person as a thief.

Environment variables:

SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY   # recommended for backend/camera service
or
SUPABASE_KEY

Optional:

STORE_VISION_PERSON_MODEL
STORE_VISION_PRODUCT_MODEL
"""

import os
import time
import uuid
import argparse
from collections import defaultdict

import cv2
from ultralytics import YOLO
from supabase import create_client


# ============================================================
# SUPABASE
# ============================================================

def create_supabase_client():
    url = os.environ.get("SUPABASE_URL")

    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
    )

    if not url:
        raise RuntimeError("SUPABASE_URL is not configured.")

    if not key:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY is not configured."
        )

    return create_client(url, key)


# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def center_of_box(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
    )


def distance_between(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return (dx * dx + dy * dy) ** 0.5


# ============================================================
# CAMERA LOADING
# ============================================================

def load_camera(sb, camera_id):
    response = (
        sb.table("cameras")
        .select("*")
        .eq("id", camera_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            f"Camera {camera_id} was not found."
        )

    camera = response.data[0]

    if not camera.get("enabled", True):
        raise RuntimeError(
            f"Camera {camera_id} is disabled."
        )

    return camera


def open_camera(camera):
    source_type = camera.get("source_type", "webcam")
    source_uri = camera.get("source_uri")

    if source_type == "webcam":
        if source_uri:
            try:
                source = int(source_uri)
            except Exception:
                source = source_uri
        else:
            source = 0

    elif source_type in ("rtsp", "file"):
        source = source_uri

    else:
        source = source_uri or 0

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open camera source: {source}"
        )

    return cap


# ============================================================
# VISIT TRACKER
# ============================================================

class VisitTracker:

    def __init__(self, sb, camera):
        self.sb = sb
        self.camera = camera

        self.owner_id = camera["owner_id"]
        self.store_id = camera["store_id"]
        self.camera_id = camera["id"]

        self.active_visits = {}

    # --------------------------------------------------------
    # Create visit
    # --------------------------------------------------------

    def create_visit(self, person_track_id):

        existing = (
            self.sb.table("store_visits")
            .select("id")
            .eq("camera_id", self.camera_id)
            .eq("person_track_id", person_track_id)
            .eq("status", "active")
            .limit(1)
            .execute()
        )

        if existing.data:
            return existing.data[0]["id"]

        payload = {
            "owner_id": self.owner_id,
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "person_track_id": person_track_id,
            "entered_at": now_iso(),
            "status": "active",
            "observed_item_count": 0,
            "billed_item_count": 0,
            "metadata": {
                "anonymous_tracking": True
            }
        }

        response = (
            self.sb.table("store_visits")
            .insert(payload)
            .execute()
        )

        if not response.data:
            return None

        visit_id = response.data[0]["id"]

        self.active_visits[person_track_id] = visit_id

        return visit_id

    # --------------------------------------------------------
    # Entry event
    # --------------------------------------------------------

    def register_entry(self, person_track_id):

        visit_id = self.create_visit(person_track_id)

        event = {
            "owner_id": self.owner_id,
            "store_id": self.store_id,
            "event_type": "entry",
            "camera_id": str(self.camera_id),
            "person_track_id": person_track_id,
            "occurred_at": now_iso(),
            "metadata": {
                "anonymous_person_id": person_track_id
            }
        }

        try:
            self.sb.table("visitor_events").insert(event).execute()
        except Exception as exc:
            print(
                "Visitor entry event error:",
                exc
            )

        return visit_id

    # --------------------------------------------------------
    # Exit event
    # --------------------------------------------------------

    def register_exit(self, person_track_id):

        visit_id = self.active_visits.get(person_track_id)

        if not visit_id:
            response = (
                self.sb.table("store_visits")
                .select("id")
                .eq("camera_id", self.camera_id)
                .eq("person_track_id", person_track_id)
                .eq("status", "active")
                .order("entered_at", desc=True)
                .limit(1)
                .execute()
            )

            if response.data:
                visit_id = response.data[0]["id"]

        if not visit_id:
            return None

        try:

            self.sb.table("visitor_events").insert({
                "owner_id": self.owner_id,
                "store_id": self.store_id,
                "event_type": "exit",
                "camera_id": str(self.camera_id),
                "person_track_id": person_track_id,
                "occurred_at": now_iso(),
                "metadata": {
                    "anonymous_person_id": person_track_id
                }
            }).execute()

        except Exception as exc:
            print(
                "Visitor exit event error:",
                exc
            )

        try:

            self.sb.table("store_visits").update({
                "exited_at": now_iso()
            }).eq(
                "id",
                visit_id
            ).execute()

        except Exception as exc:
            print(
                "Visit exit update error:",
                exc
            )

        # ----------------------------------------------------
        # Reconcile billing
        # ----------------------------------------------------

        try:

            result = self.sb.rpc(
                "fn_reconcile_visit",
                {
                    "p_visit_id": visit_id
                }
            ).execute()

            print(
                "Visit reconciliation:",
                result.data
            )

        except Exception as exc:

            print(
                "Visit reconciliation error:",
                exc
            )

        self.active_visits.pop(
            person_track_id,
            None
        )

        return visit_id


# ============================================================
# LINE CROSSING
# ============================================================

class LineCrossingDetector:

    def __init__(self, camera):

        self.x1 = safe_float(
            camera.get("line_x1"),
            0
        )

        self.y1 = safe_float(
            camera.get("line_y1"),
            0
        )

        self.x2 = safe_float(
            camera.get("line_x2"),
            0
        )

        self.y2 = safe_float(
            camera.get("line_y2"),
            0
        )

        self.previous_side = {}

    def side(self, point):

        px, py = point

        value = (
            (self.x2 - self.x1) * (py - self.y1)
            -
            (self.y2 - self.y1) * (px - self.x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    def update(self, track_id, point):

        current = self.side(point)

        previous = self.previous_side.get(
            track_id
        )

        self.previous_side[track_id] = current

        if previous is None:
            return None

        if previous == current:
            return None

        if previous < 0 and current > 0:
            return "entry"

        if previous > 0 and current < 0:
            return "exit"

        return None


# ============================================================
# PRODUCT PIPELINE
# ============================================================

class ProductPipeline:

    def __init__(self, sb, camera):

        self.sb = sb
        self.camera = camera

        self.owner_id = camera["owner_id"]
        self.store_id = camera["store_id"]
        self.camera_id = camera["id"]

        self.products_by_name = {}

        self.product_states = defaultdict(
            lambda: {
                "frames_seen": 0,
                "last_seen": 0,
                "carrying_person": None,
                "observed_item_id": None,
                "returned": False,
                "first_seen": None
            }
        )

        self.load_products()

    # --------------------------------------------------------
    # PRODUCT DATABASE
    # --------------------------------------------------------

    def load_products(self):

        response = (
            self.sb.table("products")
            .select(
                "id,name,category,sku"
            )
            .eq(
                "user_id",
                self.owner_id
            )
            .execute()
        )

        products = response.data or []

        self.products_by_name = {
            p["name"].strip().lower(): p
            for p in products
            if p.get("name")
        }

        print(
            f"Loaded {len(products)} products."
        )

        return products

    # --------------------------------------------------------
    # Match detected label to database product
    # --------------------------------------------------------

    def match_product(self, label):

        if not label:
            return None

        normalized = label.strip().lower()

        if normalized in self.products_by_name:
            return self.products_by_name[
                normalized
            ]

        # Partial matching
        for name, product in self.products_by_name.items():

            if (
                normalized in name
                or
                name in normalized
            ):
                return product

        return None

    # --------------------------------------------------------
    # Find person carrying product
    # --------------------------------------------------------

    def find_person_for_product(
        self,
        product_box,
        persons
    ):

        product_center = center_of_box(
            product_box
        )

        best_person = None
        best_distance = float("inf")

        for person in persons:

            person_box = person["box"]

            # Most reliable association:
            # product center is inside person's bbox
            if point_inside_box(
                product_center,
                person_box
            ):
                return person

            person_center = center_of_box(
                person_box
            )

            distance = distance_between(
                product_center,
                person_center
            )

            if distance < best_distance:
                best_distance = distance
                best_person = person

        # Do not associate extremely distant objects.
        if best_person is not None:

            person_box = best_person["box"]

            px1, py1, px2, py2 = person_box

            person_width = max(
                1,
                px2 - px1
            )

            person_height = max(
                1,
                py2 - py1
            )

            max_distance = max(
                person_width,
                person_height
            ) * 0.75

            if best_distance <= max_distance:
                return best_person

        return None

    # --------------------------------------------------------
    # Save observed item
    # --------------------------------------------------------

    def create_observed_item(
        self,
        visit_id,
        person_track_id,
        label,
        product_id,
        confidence,
        bbox,
        centroid
    ):

        payload = {
            "owner_id": self.owner_id,
            "store_id": self.store_id,
            "visit_id": visit_id,
            "product_id": product_id,
            "detected_label": label,
            "quantity": 1,
            "confidence": confidence,
            "picked_at": now_iso(),
            "status": "observed",
            "camera_id": self.camera_id,
            "person_track_id": person_track_id,
            "first_seen_at": now_iso(),
            "last_seen_at": now_iso(),
            "bbox": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3]
            },
            "centroid": {
                "x": centroid[0],
                "y": centroid[1]
            },
            "metadata": {
                "anonymous_tracking": True
            }
        }

        try:

            response = (
                self.sb.table("observed_items")
                .insert(payload)
                .execute()
            )

            if response.data:
                return response.data[0]["id"]

        except Exception as exc:

            print(
                "Observed item insert error:",
                exc
            )

        return None

    # --------------------------------------------------------
    # Update observed item
    # --------------------------------------------------------

    def update_observed_item(
        self,
        observed_item_id,
        bbox,
        centroid
    ):

        try:

            self.sb.table(
                "observed_items"
            ).update({
                "last_seen_at": now_iso(),
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3]
                },
                "centroid": {
                    "x": centroid[0],
                    "y": centroid[1]
                }
            }).eq(
                "id",
                observed_item_id
            ).execute()

        except Exception as exc:

            print(
                "Observed item update error:",
                exc
            )

    # --------------------------------------------------------
    # Mark returned
    # --------------------------------------------------------

    def mark_returned(
        self,
        observed_item_id
    ):

        try:

            self.sb.table(
                "observed_items"
            ).update({
                "status": "returned",
                "returned_at": now_iso(),
                "last_seen_at": now_iso()
            }).eq(
                "id",
                observed_item_id
            ).execute()

        except Exception as exc:

            print(
                "Return update error:",
                exc
            )

    # --------------------------------------------------------
    # Attraction event
    # --------------------------------------------------------

    def create_attraction_event(
        self,
        visit_id,
        person_track_id,
        label,
        product_id,
        event_type,
        confidence,
        duration_seconds=None
    ):

        payload = {
            "owner_id": self.owner_id,
            "store_id": self.store_id,
            "visit_id": visit_id,
            "product_id": product_id,
            "detected_label": label,
            "person_track_id": person_track_id,
            "event_type": event_type,
            "started_at": now_iso(),
            "confidence": confidence,
            "metadata": {
                "anonymous_tracking": True
            }
        }

        if duration_seconds is not None:
            payload["duration_seconds"] = (
                duration_seconds
            )

        try:

            self.sb.table(
                "retail_attraction_events"
            ).insert(payload).execute()

        except Exception as exc:

            print(
                "Attraction event error:",
                exc
            )

    # --------------------------------------------------------
    # Process products
    # --------------------------------------------------------

    def process_products(
        self,
        product_results,
        persons,
        visit_tracker,
        frame_number
    ):

        if not product_results:
            return []

        result = product_results[0]

        if result.boxes is None:
            return []

        detections = []

        boxes = result.boxes

        names = result.names

        for index in range(
            len(boxes)
        ):

            box_data = boxes[index]

            xyxy = box_data.xyxy[0].tolist()

            x1, y1, x2, y2 = [
                float(v)
                for v in xyxy
            ]

            confidence = safe_float(
                box_data.conf[0].item()
            )

            class_id = int(
                box_data.cls[0].item()
            )

            label = names.get(
                class_id,
                str(class_id)
            )

            track_id = None

            if getattr(
                boxes,
                "id",
                None
            ) is not None:

                try:
                    track_id = int(
                        boxes.id[index].item()
                    )
                except Exception:
                    track_id = None

            if track_id is None:

                track_id = (
                    f"det-{index}"
                )

            product_key = str(
                track_id
            )

            centroid = center_of_box(
                (x1, y1, x2, y2)
            )

            person = self.find_person_for_product(
                (x1, y1, x2, y2),
                persons
            )

            person_id = None
            visit_id = None

            if person:

                person_id = person[
                    "person_id"
                ]

                visit_id = visit_tracker.active_visits.get(
                    person_id
                )

                if visit_id is None:

                    visit_id = (
                        visit_tracker.register_entry(
                            person_id
                        )
                    )

            state = self.product_states[
                product_key
            ]

            state["frames_seen"] += 1
            state["last_seen"] = frame_number

            if state["first_seen"] is None:
                state["first_seen"] = frame_number

            # ------------------------------------------------
            # Product has a person association
            # ------------------------------------------------

            if person_id and visit_id:

                state["carrying_person"] = person_id

                product = self.match_product(
                    label
                )

                product_id = (
                    product["id"]
                    if product
                    else None
                )

                # --------------------------------------------
                # Confirm pickup
                # --------------------------------------------

                if (
                    state["observed_item_id"]
                    is None
                    and
                    state["frames_seen"] >= 3
                ):

                    observed_item_id = (
                        self.create_observed_item(
                            visit_id=visit_id,
                            person_track_id=person_id,
                            label=label,
                            product_id=product_id,
                            confidence=confidence,
                            bbox=(
                                x1,
                                y1,
                                x2,
                                y2
                            ),
                            centroid=centroid
                        )
                    )

                    state[
                        "observed_item_id"
                    ] = observed_item_id

                    if observed_item_id:

                        self.create_attraction_event(
                            visit_id=visit_id,
                            person_track_id=person_id,
                            label=label,
                            product_id=product_id,
                            event_type="pick",
                            confidence=confidence
                        )

                # --------------------------------------------
                # Existing carried item
                # --------------------------------------------

                elif state["observed_item_id"]:

                    self.update_observed_item(
                        state[
                            "observed_item_id"
                        ],
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        centroid
                    )

                    self.create_attraction_event(
                        visit_id=visit_id,
                        person_track_id=person_id,
                        label=label,
                        product_id=product_id,
                        event_type="carry",
                        confidence=confidence
                    )

            detections.append({
                "track_id": track_id,
                "label": label,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "centroid": centroid,
                "person_id": person_id
            })

        # ----------------------------------------------------
        # Detect returned products
        # ----------------------------------------------------

        current_tracks = {
            str(d["track_id"])
            for d in detections
        }

        for track_id, state in list(
            self.product_states.items()
        ):

            if (
                state["observed_item_id"]
                and
                state["last_seen"] <
                frame_number - 15
                and
                not state["returned"]
            ):

                self.mark_returned(
                    state["observed_item_id"]
                )

                state["returned"] = True

        return detections


# ============================================================
# PERSON TRACKING
# ============================================================

class PersonPipeline:

    def __init__(
        self,
        sb,
        camera,
        model
    ):

        self.sb = sb
        self.camera = camera
        self.model = model

        self.owner_id = camera["owner_id"]

        self.camera_id = str(
            camera["id"]
        )

        self.person_ids = {}

    def anonymous_person_id(
        self,
        tracker_id
    ):

        key = str(
            tracker_id
        )

        if key not in self.person_ids:

            self.person_ids[key] = (
                f"{self.camera_id}-person-{key}"
            )

        return self.person_ids[key]

    def process(
        self,
        frame
    ):

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=safe_float(
                self.camera.get(
                    "confidence",
                    0.35
                ),
                0.35
            ),
            verbose=False
        )

        persons = []

        if not results:
            return persons, results

        result = results[0]

        if result.boxes is None:
            return persons, results

        names = result.names
        boxes = result.boxes

        for index in range(
            len(boxes)
        ):

            cls_id = int(
                boxes.cls[index].item()
            )

            label = names.get(
                cls_id,
                ""
            )

            if label.lower() != "person":
                continue

            confidence = safe_float(
                boxes.conf[index].item()
            )

            xyxy = boxes.xyxy[
                index
            ].tolist()

            x1, y1, x2, y2 = [
                float(v)
                for v in xyxy
            ]

            tracker_id = None

            if getattr(
                boxes,
                "id",
                None
            ) is not None:

                try:

                    tracker_id = int(
                        boxes.id[index].item()
                    )

                except Exception:

                    tracker_id = None

            if tracker_id is None:
                tracker_id = index

            person_id = (
                self.anonymous_person_id(
                    tracker_id
                )
            )

            persons.append({
                "tracker_id": tracker_id,
                "person_id": person_id,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "centroid": center_of_box(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            })

        return persons, results


# ============================================================
# DRAWING
# ============================================================

def draw_persons(
    frame,
    persons
):

    for person in persons:

        x1, y1, x2, y2 = [
            int(v)
            for v in person["box"]
        ]

        label = (
            person["person_id"]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )


def draw_products(
    frame,
    detections
):

    for detection in detections:

        x1, y1, x2, y2 = [
            int(v)
            for v in detection["box"]
        ]

        label = detection["label"]

        confidence = detection[
            "confidence"
        ]

        text = (
            f"{label} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


def draw_line(
    frame,
    camera
):

    try:

        x1 = int(
            safe_float(
                camera.get("line_x1")
            )
        )

        y1 = int(
            safe_float(
                camera.get("line_y1")
            )
        )

        x2 = int(
            safe_float(
                camera.get("line_x2")
            )
        )

        y2 = int(
            safe_float(
                camera.get("line_y2")
            )
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

    except Exception:
        pass


# ============================================================
# LOAD MODELS
# ============================================================

def load_models(camera):

    person_model_path = (
        os.environ.get(
            "STORE_VISION_PERSON_MODEL"
        )
        or
        "yolov8n.pt"
    )

    product_model_path = (
        os.environ.get(
            "STORE_VISION_PRODUCT_MODEL"
        )
        or
        "models/store_products_seg.pt"
    )

    camera_model = camera.get(
        "model_path"
    )

    # If a specific camera model exists,
    # use it as person model only if
    # environment variable was not provided.
    if (
        not os.environ.get(
            "STORE_VISION_PERSON_MODEL"
        )
        and camera_model
        and os.path.exists(
            camera_model
        )
    ):
        person_model_path = camera_model

    print(
        "Loading person model:",
        person_model_path
    )

    person_model = YOLO(
        person_model_path
    )

    if not os.path.exists(
        product_model_path
    ):

        raise FileNotFoundError(
            "Product model was not found:\n"
            f"{product_model_path}\n\n"
            "Set STORE_VISION_PRODUCT_MODEL "
            "to your trained product model."
        )

    print(
        "Loading product model:",
        product_model_path
    )

    product_model = YOLO(
        product_model_path
    )

    return (
        person_model,
        product_model
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    camera_id,
    display=True
):

    print()
    print("=" * 60)
    print("STORE VISION AI")
    print("Camera Pipeline Starting")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Supabase
    # --------------------------------------------------------

    sb = create_supabase_client()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = load_camera(
        sb,
        camera_id
    )

    print(
        "Camera:",
        camera.get("name")
    )

    print(
        "Store:",
        camera.get("store_id")
    )

    print(
        "Owner:",
        camera.get("owner_id")
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    person_model, product_model = (
        load_models(camera)
    )

    # --------------------------------------------------------
    # Camera source
    # --------------------------------------------------------

    cap = open_camera(
        camera
    )

    # --------------------------------------------------------
    # Pipelines
    # --------------------------------------------------------

    visit_tracker = VisitTracker(
        sb,
        camera
    )

    line_detector = LineCrossingDetector(
        camera
    )

    person_pipeline = PersonPipeline(
        sb,
        camera,
        person_model
    )

    product_pipeline = ProductPipeline(
        sb,
        camera
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    frame_number = 0

    print()
    print("Pipeline is running.")
    print("Press Q to stop.")
    print()

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "Unable to read frame."
                )

                time.sleep(0.1)
                continue

            frame_number += 1

            # ------------------------------------------------
            # Resize if camera dimensions are configured
            # ------------------------------------------------

            configured_width = camera.get(
                "frame_width"
            )

            configured_height = camera.get(
                "frame_height"
            )

            if (
                configured_width
                and
                configured_height
            ):

                try:

                    frame = cv2.resize(
                        frame,
                        (
                            int(
                                configured_width
                            ),
                            int(
                                configured_height
                            )
                        )
                    )

                except Exception:
                    pass

            # ------------------------------------------------
            # PERSON DETECTION
            # ------------------------------------------------

            persons, person_results = (
                person_pipeline.process(
                    frame
                )
            )

            # ------------------------------------------------
            # ENTRY / EXIT DETECTION
            # ------------------------------------------------

            for person in persons:

                person_id = (
                    person["person_id"]
                )

                event = line_detector.update(
                    person_id,
                    person["centroid"]
                )

                if event == "entry":

                    visit_tracker.register_entry(
                        person_id
                    )

                    print(
                        f"[ENTRY] {person_id}"
                    )

                elif event == "exit":

                    visit_id = (
                        visit_tracker.register_exit(
                            person_id
                        )
                    )

                    print(
                        f"[EXIT] "
                        f"{person_id} "
                        f"visit={visit_id}"
                    )

            # ------------------------------------------------
            # PRODUCT DETECTION/TRACKING
            # ------------------------------------------------

            try:

                product_results = (
                    product_model.track(
                        frame,
                        persist=True,
                        conf=safe_float(
                            camera.get(
                                "confidence",
                                0.35
                            ),
                            0.35
                        ),
                        iou=0.5,

                        # IMPORTANT:
                        # Supports many objects.
                        # Does NOT guarantee detection
                        # of 27 objects unless the model
                        # itself can detect them.
                        max_det=1000,

                        verbose=False
                    )
                )

                product_detections = (
                    product_pipeline.process_products(
                        product_results,
                        persons,
                        visit_tracker,
                        frame_number
                    )
                )

            except Exception as exc:

                print(
                    "Product detection error:",
                    exc
                )

                product_detections = []

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            if display:

                draw_persons(
                    frame,
                    persons
                )

                draw_products(
                    frame,
                    product_detections
                )

                draw_line(
                    frame,
                    camera
                )

                status_text = (
                    f"Persons: {len(persons)} | "
                    f"Products: "
                    f"{len(product_detections)}"
                )

                cv2.putText(
                    frame,
                    status_text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    "Store Vision AI",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q")
                ):
                    break

    except KeyboardInterrupt:

        print()
        print(
            "Pipeline stopped."
        )

    finally:

        cap.release()

        if display:
            cv2.destroyAllWindows()

        print(
            "Camera released."
        )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Store Vision AI Camera Pipeline"
        )
    )

    parser.add_argument(
        "--camera-id",
        required=True,
        help=(
            "UUID of the camera "
            "from the cameras table"
        )
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Run without opening "
            "the OpenCV display window"
        )
    )

    args = parser.parse_args()

    run_pipeline(
        camera_id=args.camera_id,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()