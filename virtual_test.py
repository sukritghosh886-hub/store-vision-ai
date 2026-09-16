import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import store_events
import video_pipeline


@dataclass
class Result:
    data: list


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table_name = table
        self.filters = []
        self.operation = "select"
        self.payload = None

    def select(self, *_):
        self.operation = "select"
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = self.client.rows.setdefault(self.table_name, [])

        if self.operation == "insert":
            row = dict(self.payload)
            row.setdefault(
                "id",
                f"{self.table_name}-{len(rows) + 1}",
            )
            rows.append(row)
            return Result([row])

        if self.operation == "update":
            for row in rows:
                if all(
                    row.get(k) == v
                    for k, v in self.filters
                ):
                    row.update(self.payload)

            return Result(rows)

        data = [
            row
            for row in rows
            if all(
                row.get(k) == v
                for k, v in self.filters
            )
        ]

        return Result(data)


class FakeClient:
    def __init__(self):
        self.rows = {
            "visits": [],
            "item_events": [],
            "billing_events": [],
            "alerts": [],
        }

    def table(self, name):
        return FakeQuery(self, name)


def make_video(path):
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (640, 400),
    )

    assert writer.isOpened(), \
        "Synthetic video writer failed"

    for _ in range(5):
        frame = np.zeros(
            (400, 640, 3),
            dtype=np.uint8,
        )

        frame[:] = 180
        writer.write(frame)

    writer.release()


def test_tracker_and_helpers():
    tracker = video_pipeline.ItemTracker(
        max_distance=50,
        max_missing=2,
    )

    first = tracker.update([
        {
            "class_name": "book",
            "bbox": [90, 90, 110, 110],
        }
    ])

    second = tracker.update([
        {
            "class_name": "book",
            "bbox": [95, 95, 115, 115],
        }
    ])

    assert list(first) == [1]
    assert list(second) == [1]

    assert (
        video_pipeline.calculate_zone(
            {"bbox": [90, 260, 110, 280]},
            400,
            0.55,
        )
        == "shelf"
    )

    assert (
        video_pipeline.calculate_zone(
            {"bbox": [90, 40, 110, 60]},
            400,
            0.55,
        )
        == "store"
    )


def test_store_events_billing_alert():
    fake = FakeClient()

    original_client = store_events._client
    store_events._client = lambda: fake

    try:
        visit = store_events.start_visit(
            "store-1",
            1,
        )

        store_events.log_item_event(
            visit,
            "book",
            "shelf",
            0.95,
        )

        assert (
            store_events.get_unpaid_count(visit)
            == 1
        )

        alert = store_events.close_visit(
            visit,
            "store-1",
        )

        assert alert
        assert (
            alert["alert_type"]
            == "unpaid_item_flag"
        )

        assert (
            fake.rows["visits"][0]["status"]
            == "exited_flagged"
        )

    finally:
        store_events._client = original_client


def test_video_pipeline_synthetic():
    fake = FakeClient()

    original_client = store_events._client
    original_detect = video_pipeline.detect_objects

    store_events._client = lambda: fake

    calls = {"n": 0}

    def fake_detect(frame, confidence=0.4):
        calls["n"] += 1

        y = min(
            100 + (calls["n"] - 1) * 80,
            330,
        )

        person = {
            "class_name": "person",
            "bbox": [
                80,
                y,
                140,
                y + 60,
            ],
            "confidence": 0.99,
        }

        book = {
            "class_name": "book",
            "bbox": [
                90,
                260,
                140,
                320,
            ],
            "confidence": 0.95,
        }

        return frame.copy(), [
            person,
            book,
        ]

    video_pipeline.detect_objects = fake_detect

    path = None

    try:
        fd, path = tempfile.mkstemp(
            suffix=".mp4"
        )

        os.close(fd)

        make_video(path)

        frames = list(
            video_pipeline.process_video(
                path,
                "store-1",
                frame_stride=1,
            )
        )

        assert len(frames) == 5

        assert fake.rows["visits"], \
            "visit was not created"

        assert fake.rows["item_events"], \
            "item event was not created"

        assert fake.rows["alerts"], \
            "unpaid-item alert was not created"

        assert (
            fake.rows["visits"][0]["status"]
            == "exited_flagged"
        )

    finally:
        video_pipeline.detect_objects = original_detect
        store_events._client = original_client

        if path and Path(path).exists():
            Path(path).unlink()


def test_api_contract():
    import main

    original_process = main.process_video

    main.process_video = lambda *args, **kwargs: iter([
        {
            "detections": [
                {"class_name": "person"}
            ],
            "people": [
                {"track_id": 1}
            ],
            "items": [
                {"track_id": 2}
            ],
            "exits": [],
            "alerts": [],
        }
    ])

    try:
        client = TestClient(main.app)

        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/stores").status_code == 200
        assert client.get("/alerts").status_code == 200

        response = client.post(
            "/process-video"
        )

        assert response.status_code == 422

    finally:
        main.process_video = original_process


if __name__ == "__main__":
    tests = [
        test_tracker_and_helpers,
        test_store_events_billing_alert,
        test_video_pipeline_synthetic,
        test_api_contract,
    ]

    for test in tests:
        test()
        print("PASS", test.__name__)

    print("VIRTUAL_TESTS_COMPLETE")