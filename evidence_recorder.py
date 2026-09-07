"""
Store Vision AI
Evidence Recorder

Saves security evidence locally and uploads it to
Supabase Storage.

Evidence is associated with:
    store
    visit
    camera
    anonymous person_track_id
    observed item

No facial recognition is performed.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2

try:
    from supabase import create_client
except ImportError:
    create_client = None


DEFAULT_EVIDENCE_DIR = os.getenv(
    "STORE_VISION_EVIDENCE_DIR",
    "evidence",
)

DEFAULT_BUCKET = os.getenv(
    "SUPABASE_EVIDENCE_BUCKET",
    "security-evidence",
)


class EvidenceRecorder:

    def __init__(
        self,
        evidence_dir: str = DEFAULT_EVIDENCE_DIR,
        supabase_client: Any = None,
        bucket: str = DEFAULT_BUCKET,
    ):
        self.root = Path(evidence_dir)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.bucket = bucket
        self.supabase = supabase_client

        if self.supabase is None:
            self.supabase = self._create_supabase_client()

    # ========================================================
    # SUPABASE CLIENT
    # ========================================================

    def _create_supabase_client(self):
        if create_client is None:
            return None

        url = os.getenv("SUPABASE_URL")

        # The camera/security pipeline should normally use
        # the service-role key because evidence is server-side.
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not key:
            key = os.getenv("SUPABASE_KEY")

        if not key:
            key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

        if not url or not key:
            return None

        try:
            return create_client(url, key)
        except Exception:
            return None

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
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:

        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"{timestamp}_"
            f"{event_type}_"
            f"{unique_id}.jpg"
        )

        directory = self.root / str(store_id)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_path = directory / filename

        success = cv2.imwrite(
            str(local_path),
            frame,
        )

        if not success:
            raise RuntimeError(
                "Could not save evidence snapshot."
            )

        storage_path = self._build_storage_path(
            store_id=store_id,
            filename=filename,
        )

        self._upload_file(
            local_path=local_path,
            storage_path=storage_path,
            content_type="image/jpeg",
        )

        self._insert_evidence_record(
            store_id=store_id,
            visit_id=visit_id,
            observed_item_id=observed_item_id,
            camera_id=camera_id,
            person_track_id=person_track_id,
            evidence_type="snapshot",
            storage_path=storage_path,
            metadata={
                "event_type": event_type,
                **(metadata or {}),
            },
        )

        return storage_path

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
        visit_id: Optional[str] = None,
        person_track_id: Optional[str] = None,
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:

        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"{timestamp}_"
            f"{event_type}_"
            f"{unique_id}.mp4"
        )

        directory = self.root / str(store_id)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_path = directory / filename

        width, height = frame_size

        if fps is None or fps <= 0:
            fps = 15.0

        writer = cv2.VideoWriter(
            str(local_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create evidence video."
            )

        frame_count = 0

        try:
            for frame in frames:
                if frame is None:
                    continue

                writer.write(frame)
                frame_count += 1

        finally:
            writer.release()

        if frame_count == 0:
            raise RuntimeError(
                "No frames were available for evidence video."
            )

        storage_path = self._build_storage_path(
            store_id=store_id,
            filename=filename,
        )

        self._upload_file(
            local_path=local_path,
            storage_path=storage_path,
            content_type="video/mp4",
        )

        self._insert_evidence_record(
            store_id=store_id,
            visit_id=visit_id,
            observed_item_id=observed_item_id,
            camera_id=camera_id,
            person_track_id=person_track_id,
            evidence_type="video_clip",
            storage_path=storage_path,
            metadata={
                "event_type": event_type,
                "fps": float(fps),
                "frame_count": frame_count,
                "frame_width": int(width),
                "frame_height": int(height),
                **(metadata or {}),
            },
        )

        return storage_path

    # ========================================================
    # STORAGE PATH
    # ========================================================

    def _build_storage_path(
        self,
        store_id: str,
        filename: str,
    ) -> str:

        # Important:
        # The first path segment is the store UUID.
        #
        # This matches the Storage RLS policy that restricts
        # users to evidence belonging to their own store.

        return (
            f"{store_id}/"
            f"{filename}"
        )

    # ========================================================
    # UPLOAD
    # ========================================================

    def _upload_file(
        self,
        local_path: Path,
        storage_path: str,
        content_type: str,
    ) -> bool:

        if self.supabase is None:
            # Local-only mode is intentionally supported.
            # This allows offline/local deployments.
            return False

        try:

            with open(
                local_path,
                "rb",
            ) as file_handle:

                file_bytes = file_handle.read()

            self.supabase.storage \
                .from_(self.bucket) \
                .upload(
                    storage_path,
                    file_bytes,
                    {
                        "content-type": content_type,
                        "upsert": "false",
                    },
                )

            return True

        except Exception as exc:

            # Do not crash the camera pipeline merely because
            # cloud storage is temporarily unavailable.
            print(
                "Evidence upload failed:",
                exc,
            )

            return False

    # ========================================================
    # DATABASE RECORD
    # ========================================================

    def _insert_evidence_record(
        self,
        store_id: str,
        visit_id: Optional[str],
        observed_item_id: Optional[str],
        camera_id: Optional[str],
        person_track_id: Optional[str],
        evidence_type: str,
        storage_path: str,
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:

        if self.supabase is None:
            return None

        payload = {
            "store_id": str(store_id),
            "visit_id": (
                str(visit_id)
                if visit_id
                else None
            ),
            "observed_item_id": (
                str(observed_item_id)
                if observed_item_id
                else None
            ),
            "camera_id": (
                str(camera_id)
                if camera_id
                else None
            ),
            "person_track_id": (
                str(person_track_id)
                if person_track_id is not None
                else None
            ),
            "evidence_type": evidence_type,
            "storage_path": storage_path,
            "metadata": metadata or {},
        }

        try:

            response = (
                self.supabase
                .table("security_evidence")
                .insert(payload)
                .execute()
            )

            if response.data:
                return response.data[0]

            return None

        except Exception as exc:

            print(
                "Could not create security_evidence "
                "database record:",
                exc,
            )

            return None

    # ========================================================
    # SNAPSHOT + DATABASE RECORD WITHOUT UPLOAD
    # ========================================================

    def record_event_frame(
        self,
        frame,
        store_id: str,
        visit_id: Optional[str],
        person_track_id: Optional[str],
        event_type: str,
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:

        return self.save_snapshot(
            frame=frame,
            store_id=store_id,
            visit_id=visit_id,
            person_track_id=person_track_id,
            event_type=event_type,
            observed_item_id=observed_item_id,
            camera_id=camera_id,
            metadata=metadata,
        )

    # ========================================================
    # STORAGE URL
    # ========================================================

    def create_signed_url(
        self,
        storage_path: str,
        expires_in: int = 3600,
    ) -> Optional[str]:

        if self.supabase is None:
            return None

        try:

            response = (
                self.supabase
                .storage
                .from_(self.bucket)
                .create_signed_url(
                    storage_path,
                    expires_in,
                )
            )

            if isinstance(response, dict):
                return response.get("signedURL") or response.get(
                    "signedUrl"
                )

            return None

        except Exception as exc:

            print(
                "Could not create evidence signed URL:",
                exc,
            )

            return None


__all__ = [
    "EvidenceRecorder",
]