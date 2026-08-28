"""
Store Vision AI — camera ingestion pipeline

Runs YOLO person detection + ByteTrack tracking on a camera feed, counts
entries/exits across the line configured on the `cameras` row, and syncs
store_visits / visitor_events to Supabase in real time.

This covers PERSON TRACKING end-to-end. Item-level detection (what someone
picks up) is a separate, harder problem — see the note at the bottom of
this file before wiring that part in.

Usage:
    python camera_pipeline.py --camera-id <uuid-from-cameras-table>

Add to requirements.txt:
    ultralytics
    supervision
    opencv-python
    supabase

Environment variables required:
    SUPABASE_URL
    SUPABASE_KEY   (service role key — this runs as a backend worker, not
                    a logged-in user, so it needs to bypass RLS policies
                    that assume an authenticated end user)
"""

import argparse
import os
import time
from datetime import datetime, timezone

import cv2
import supervision as sv
from ultralytics import YOLO
from supabase import create_client, Client

PERSON_CLASS_ID = 0  # COCO 'person' class in standard YOLO checkpoints


def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def load_camera(sb: Client, camera_id: str) -> dict:
    resp = sb.table("cameras").select("*").eq("id", camera_id).single().execute()
    if not resp.data:
        raise ValueError(f"No camera found with id={camera_id}")
    return resp.data


def open_source(camera: dict) -> cv2.VideoCapture:
    source_type = camera["source_type"]
    uri = camera["source_uri"]
    if source_type == "webcam":
        return cv2.VideoCapture(int(uri))  # source_uri holds the device index
    return cv2.VideoCapture(uri)  # rtsp:// URL or file path


class VisitTracker:
    """Tracks which person_track_ids currently have an open store_visits row,
    and writes the corresponding visitor_events rows."""

    def __init__(self, sb: Client, camera: dict):
        self.sb = sb
        self.camera = camera
        self.open_visits: dict[int, str] = {}  # tracker_id -> store_visits.id

    def on_entry(self, tracker_id: int):
        if tracker_id in self.open_visits:
            return
        row = {
            "owner_id": self.camera["owner_id"],
            "store_id": self.camera["store_id"],
            "camera_id": self.camera["id"],
            "person_track_id": str(tracker_id),
            "status": "active",
        }
        result = self.sb.table("store_visits").insert(row).execute()
        self.open_visits[tracker_id] = result.data[0]["id"]

        self.sb.table("visitor_events").insert({
            "owner_id": self.camera["owner_id"],
            "store_id": self.camera["store_id"],
            "event_type": "entry",
            "camera_id": self.camera["name"],
            "person_track_id": str(tracker_id),
        }).execute()
        print(f"[entry] track_id={tracker_id}")

    def on_exit(self, tracker_id: int):
        visit_id = self.open_visits.pop(tracker_id, None)
        if visit_id is None:
            return
        self.sb.table("store_visits").update({
            "status": "completed",
            "exited_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", visit_id).execute()

        self.sb.table("visitor_events").insert({
            "owner_id": self.camera["owner_id"],
            "store_id": self.camera["store_id"],
            "event_type": "exit",
            "camera_id": self.camera["name"],
            "person_track_id": str(tracker_id),
        }).execute()
        print(f"[exit]  track_id={tracker_id}")


def run(camera_id: str):
    sb = get_supabase()
    camera = load_camera(sb, camera_id)

    if not camera["enabled"]:
        print(f"Camera '{camera['name']}' is disabled in Supabase; exiting.")
        return

    model = YOLO(camera["model_path"])
    tracker = sv.ByteTrack()
    visits = VisitTracker(sb, camera)

    line_zone = sv.LineZone(
        start=sv.Point(camera["line_x1"], camera["line_y1"]),
        end=sv.Point(camera["line_x2"], camera["line_y2"]),
    )

    cap = open_source(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["frame_height"])

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {camera['source_uri']}")

    print(f"Running pipeline for camera '{camera['name']}' ({camera['source_type']})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if camera["source_type"] == "file":
                    break  # end of video file
                time.sleep(0.5)  # brief pause on a dropped RTSP frame, then retry
                continue

            result = model(frame, verbose=False, conf=float(camera["confidence"]))[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.class_id == PERSON_CLASS_ID]
            detections = tracker.update_with_detections(detections)

            crossed_in, crossed_out = line_zone.trigger(detections)

            for tracker_id, did_cross_in in zip(detections.tracker_id, crossed_in):
                if did_cross_in:
                    visits.on_entry(int(tracker_id))

            for tracker_id, did_cross_out in zip(detections.tracker_id, crossed_out):
                if did_cross_out:
                    visits.on_exit(int(tracker_id))

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store Vision AI camera pipeline")
    parser.add_argument("--camera-id", required=True, help="UUID of the camera row in Supabase")
    args = parser.parse_args()
    run(args.camera_id)

# ---------------------------------------------------------------------------
# NOTE on item-level detection (observed_items table):
#
# Person tracking + line crossing (above) is the well-defined half of this
# project. Detecting *which item* a tracked person picks up is a much harder,
# more open-ended CV problem, and there are a few genuinely different ways
# to approach it — worth deciding deliberately rather than guessing:
#
#   1. Heuristic zone trigger: define shelf "zones" per camera, and log an
#      observed_item row (generic label, not a specific product) whenever a
#      tracked person's bounding box dwells in a zone. Fast to build, coarse.
#   2. Custom-trained item detector: fine-tune a YOLO model on your own
#      product photos so it recognizes actual SKUs. Accurate, but needs a
#      labeled dataset and training time.
#   3. Simulated data: skip live detection for now, script some plausible
#      observed_items/billing_items rows so the rest of the pipeline (alerts,
#      billing reconciliation, Streamlit dashboards) can be built and
#      demoed end-to-end before the hardest CV problem is solved.
# ---------------------------------------------------------------------------
