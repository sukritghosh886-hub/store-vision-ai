"""
Vision pipeline for Store Vision AI.

Uses YOLOv8's built-in tracker (ByteTrack) to follow each person across
frames, watches two zones (shelf / exit), and raises a review alert when
a tracked person crosses the exit zone with items picked up near the
shelf that were never billed.

Honest scoping note: COCO (YOLO's default training set) has no
retail-SKU class, so this uses a handful of COCO object classes as
stand-ins for "carried item" (bottle, backpack, handbag, suitcase,
book, cell phone). A real deployment would fine-tune on the store's
own products. Zones are simple vertical bands of the frame rather than
calibrated polygons — enough for a single fixed demo camera angle.
"""
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import store_events

PERSON_CLASS = 0
ITEM_PROXY_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    67: "cell phone",
    73: "book",
}
LOST_TRACK_FRAMES = 45  # ~1.5s at 30fps before we treat a track as gone

_model_cache = {}


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """Cached model load — avoids re-downloading/re-initializing per frame."""
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def process_video(
    video_path: str,
    store_id: str,
    shelf_zone_frac: float = 0.55,
    exit_zone_frac: float = 0.85,
    conf: float = 0.4,
    frame_stride: int = 2,
):
    """
    Generator that processes a video and yields (annotated_frame_bgr, log_line)
    for each processed frame. `log_line` is None on frames with nothing new to
    report.

    Zones are vertical bands: a person is "at the shelf" once their box center
    crosses `shelf_zone_frac` of frame width, and "at the exit" once it crosses
    `exit_zone_frac`. Point the camera so shelves are on the far side and the
    exit is near one edge for this to make sense.
    """
    model = load_model()
    classes_of_interest = [PERSON_CLASS] + list(ITEM_PROXY_CLASSES.keys())

    track_state = defaultdict(lambda: {
        "visit_id": None,
        "seen_shelf": False,
        "last_seen_frame": 0,
        "reported_items": set(),
    })

    frame_idx = 0
    results_stream = model.track(
        source=video_path,
        classes=classes_of_interest,
        conf=conf,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for result in results_stream:
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        frame = result.orig_img.copy()
        h, w = frame.shape[:2]
        shelf_x = int(w * shelf_zone_frac)
        exit_x = int(w * exit_zone_frac)
        cv2.line(frame, (shelf_x, 0), (shelf_x, h), (80, 200, 255), 2)
        cv2.line(frame, (exit_x, 0), (exit_x, h), (60, 60, 255), 2)
        cv2.putText(frame, "SHELF ZONE", (shelf_x + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 2)
        cv2.putText(frame, "EXIT ZONE", (exit_x + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)

        log_line = None
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            yield frame, log_line
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        person_boxes = [(ids[i], xyxy[i]) for i in range(len(ids)) if cls[i] == PERSON_CLASS]
        item_boxes = [(ITEM_PROXY_CLASSES[cls[i]], xyxy[i], confs[i]) for i in range(len(ids)) if cls[i] in ITEM_PROXY_CLASSES]

        active_ids = set()
        for track_id, box in person_boxes:
            active_ids.add(track_id)
            state = track_state[track_id]
            state["last_seen_frame"] = frame_idx
            cx, cy = _box_center(box)
            x1, y1, x2, y2 = box.astype(int)

            if state["visit_id"] is None:
                state["visit_id"] = store_events.start_visit(store_id, int(track_id))
                log_line = f"Track #{track_id}: new visit started"

            in_shelf = cx >= shelf_x and cx < exit_x
            in_exit = cx >= exit_x

            box_color = (0, 200, 0)

            if in_shelf and not state["seen_shelf"]:
                state["seen_shelf"] = True

            if in_shelf:
                # Any item-class box overlapping this person -> log a pickup event once per item type
                for label, ibox, iconf in item_boxes:
                    if _iou(box, ibox) > 0.02 and label not in state["reported_items"]:
                        store_events.log_item_event(state["visit_id"], label, "shelf", float(iconf))
                        state["reported_items"].add(label)
                        log_line = f"Track #{track_id}: picked up '{label}' near shelf"

            if in_exit and state["visit_id"] is not None:
                alert = store_events.close_visit(state["visit_id"], store_id)
                if alert:
                    box_color = (0, 0, 255)
                    log_line = (
                        f"Track #{track_id}: exited with {alert['unpaid_item_count']} "
                        f"unpaid item(s) — flagged for review"
                    )
                else:
                    log_line = f"Track #{track_id}: exited clean"
                state["visit_id"] = None  # ready for a fresh visit if this id re-enters later
                state["seen_shelf"] = False
                state["reported_items"] = set()

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"#{track_id}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        for label, ibox, iconf in item_boxes:
            x1, y1, x2, y2 = ibox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

        # Close out tracks that vanished without crossing the exit line (walked off-frame, occlusion, etc.)
        for track_id, state in list(track_state.items()):
            if track_id in active_ids or state["visit_id"] is None:
                continue
            if frame_idx - state["last_seen_frame"] > LOST_TRACK_FRAMES:
                alert = store_events.close_visit(state["visit_id"], store_id)
                if alert:
                    log_line = (
                        f"Track #{track_id}: lost track with {alert['unpaid_item_count']} "
                        f"unpaid item(s) — flagged for review"
                    )
                state["visit_id"] = None

        yield frame, log_line
