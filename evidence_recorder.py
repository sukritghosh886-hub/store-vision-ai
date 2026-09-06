"""
Store Vision AI
Evidence Recorder

Records snapshots and short video clips for events that require
human review.

No facial recognition is performed here.

Evidence is associated with:
    store
    visit
    anonymous person_track_id
    observed item
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2


DEFAULT_EVIDENCE_DIR = os.getenv(
    "STORE_VISION_EVIDENCE_DIR",
    "evidence",
)


class EvidenceRecorder:

    def __init__(
        self,
        evidence_dir: str = DEFAULT_EVIDENCE_DIR,
    ):

        self.root = Path(
            evidence_dir
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # SNAPSHOT
    # ========================================================

    def save_snapshot(
        self,
        frame,
        store_id: str,
        visit_id: Optional[str],
        person_track_id: Optional[str],
        event_type: str,
    ) -> str:

        timestamp = int(
            time.time()
        )

        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"{timestamp}_"
            f"{event_type}_"
            f"{unique_id}.jpg"
        )

        directory = (
            self.root
            /
            str(store_id)
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        path = (
            directory
            /
            filename
        )

        success = cv2.imwrite(
            str(path),
            frame
        )

        if not success:

            raise RuntimeError(
                "Could not save evidence snapshot."
            )

        return str(path)


    # ========================================================
    # SHORT VIDEO CLIP
    # ========================================================

    def save_video_clip(
        self,
        frames,
        fps: float,
        frame_size,
        store_id: str,
        event_type: str,
    ) -> str:

        timestamp = int(
            time.time()
        )

        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"{timestamp}_"
            f"{event_type}_"
            f"{unique_id}.mp4"
        )

        directory = (
            self.root
            /
            str(store_id)
        )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        path = (
            directory
            /
            filename
        )

        width, height = frame_size

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (width, height),
        )

        try:

            for frame in frames:

                writer.write(
                    frame
                )

        finally:

            writer.release()

        return str(path)


__all__ = [
    "EvidenceRecorder",
]