"""
Store Vision AI
Security Evidence Recorder

This module captures security evidence such as:
    - JPEG snapshots
    - MP4 video clips

It stores evidence locally and, when Supabase is configured,
uploads it to the private `security-evidence` bucket.

It also creates a corresponding row in:
    public.security_evidence

Important:
    - No facial recognition is performed.
    - People are identified only by anonymous person_track_id.
    - Evidence is connected to store, visit, camera and observed item.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2


class EvidenceRecorder:
    """
    Records security evidence for Store Vision AI.

    Supabase is optional.

    If Supabase is unavailable, evidence is still saved locally.
    """

    def __init__(
        self,
        supabase_client=None,
        bucket_name: str = "security-evidence",
        evidence_dir: str = "evidence",
    ):
        self.supabase = supabase_client
        self.bucket_name = bucket_name
        self.evidence_dir = Path(evidence_dir)

        self.evidence_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # SUPABASE CLIENT
    # ============================================================

    def _create_supabase_client(self):
        """
        Create a Supabase client only when one was not supplied.
        """

        if self.supabase is not None:
            return self.supabase

        try:
            from supabase import create_client

            supabase_url = os.getenv("SUPABASE_URL")

            supabase_key = (
                os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                or os.getenv("SUPABASE_KEY")
                or os.getenv("SUPABASE_PUBLISHABLE_KEY")
            )

            if not supabase_url or not supabase_key:
                return None

            self.supabase = create_client(
                supabase_url,
                supabase_key,
            )

            return self.supabase

        except Exception as exc:
            print(
                "[EVIDENCE] Could not create Supabase client:",
                exc,
            )
            return None

    # ============================================================
    # TIME
    # ============================================================

    @staticmethod
    def _now_iso() -> str:
        """
        Return current UTC timestamp in ISO format.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ============================================================
    # SAFE PATH
    # ============================================================

    @staticmethod
    def _safe_component(value: Optional[str]) -> str:
        """
        Make a value safe for a storage path.
        """

        if value is None:
            return "unknown"

        value = str(value).strip()

        if not value:
            return "unknown"

        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "-_."
        )

        return "".join(
            character
            if character in allowed
            else "_"
            for character in value
        )

    # ============================================================
    # SNAPSHOT
    # ============================================================

    def save_snapshot(
        self,
        frame,
        store_id: str,
        visit_id: Optional[str] = None,
        person_track_id: Optional[str] = None,
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save a JPEG security snapshot.

        Returns:
            Supabase storage path when upload succeeds,
            local path when Supabase is unavailable,
            None when saving fails.
        """

        if frame is None:
            print("[EVIDENCE] Snapshot frame is empty.")
            return None

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")

        unique_id = uuid.uuid4().hex

        filename = (
            f"security_snapshot_"
            f"{timestamp}_"
            f"{unique_id}.jpg"
        )

        local_path = (
            self.evidence_dir /
            filename
        )

        try:
            success = cv2.imwrite(
                str(local_path),
                frame,
            )

            if not success:
                print(
                    "[EVIDENCE] Failed to save snapshot:",
                    local_path,
                )
                return None

        except Exception as exc:
            print(
                "[EVIDENCE] Snapshot save error:",
                exc,
            )
            return None

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        evidence_metadata = dict(
            metadata or {}
        )

        evidence_metadata.update({
            "anonymous_tracking": True,
            "captured_at": self._now_iso(),
            "local_path": str(local_path),
        })

        # --------------------------------------------------------
        # Upload
        # --------------------------------------------------------

        storage_path = self._upload_file(
            local_path=local_path,
            store_id=store_id,
            filename=filename,
            content_type="image/jpeg",
        )

        # --------------------------------------------------------
        # Database record
        # --------------------------------------------------------

        if storage_path:

            record_created = (
                self._insert_evidence_record(
                    owner_id=owner_id,
                    store_id=store_id,
                    visit_id=visit_id,
                    observed_item_id=observed_item_id,
                    camera_id=camera_id,
                    person_track_id=person_track_id,
                    evidence_type="snapshot",
                    storage_path=storage_path,
                    captured_at=self._now_iso(),
                    metadata=evidence_metadata,
                )
            )

            if record_created:
                return storage_path

        # --------------------------------------------------------
        # Local fallback
        # --------------------------------------------------------

        print(
            "[EVIDENCE] Snapshot saved locally:",
            str(local_path),
        )

        return str(local_path)

    # ============================================================
    # VIDEO CLIP
    # ============================================================

    def save_video_clip(
        self,
        frames,
        fps: float,
        store_id: str,
        visit_id: Optional[str] = None,
        person_track_id: Optional[str] = None,
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save an MP4 security video clip.

        `frames` should contain OpenCV BGR frames.
        """

        if not frames:
            print("[EVIDENCE] No video frames supplied.")
            return None

        if fps is None or fps <= 0:
            fps = 20.0

        first_frame = frames[0]

        if first_frame is None:
            print("[EVIDENCE] First video frame is empty.")
            return None

        height, width = first_frame.shape[:2]

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")

        unique_id = uuid.uuid4().hex

        filename = (
            f"security_clip_"
            f"{timestamp}_"
            f"{unique_id}.mp4"
        )

        local_path = (
            self.evidence_dir /
            filename
        )

        writer = None

        try:
            writer = cv2.VideoWriter(
                str(local_path),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                float(fps),
                (width, height),
            )

            if not writer.isOpened():
                print(
                    "[EVIDENCE] Could not open video writer."
                )
                return None

            for frame in frames:

                if frame is None:
                    continue

                if (
                    frame.shape[1] != width
                    or frame.shape[0] != height
                ):
                    frame = cv2.resize(
                        frame,
                        (width, height),
                    )

                writer.write(frame)

        except Exception as exc:

            print(
                "[EVIDENCE] Video creation error:",
                exc,
            )

            return None

        finally:

            if writer is not None:
                writer.release()

        evidence_metadata = dict(
            metadata or {}
        )

        evidence_metadata.update({
            "anonymous_tracking": True,
            "captured_at": self._now_iso(),
            "local_path": str(local_path),
            "fps": float(fps),
            "frame_count": len(frames),
        })

        storage_path = self._upload_file(
            local_path=local_path,
            store_id=store_id,
            filename=filename,
            content_type="video/mp4",
        )

        if storage_path:

            record_created = (
                self._insert_evidence_record(
                    owner_id=owner_id,
                    store_id=store_id,
                    visit_id=visit_id,
                    observed_item_id=observed_item_id,
                    camera_id=camera_id,
                    person_track_id=person_track_id,
                    evidence_type="video_clip",
                    storage_path=storage_path,
                    captured_at=self._now_iso(),
                    metadata=evidence_metadata,
                )
            )

            if record_created:
                return storage_path

        print(
            "[EVIDENCE] Video saved locally:",
            str(local_path),
        )

        return str(local_path)

    # ============================================================
    # UPLOAD
    # ============================================================

    def _upload_file(
        self,
        local_path: Path,
        store_id: str,
        filename: str,
        content_type: str,
    ) -> Optional[str]:
        """
        Upload evidence to the private Supabase bucket.

        Storage structure:

            security-evidence/
                <store_id>/
                    <filename>
        """

        sb = self._create_supabase_client()

        if sb is None:
            print(
                "[EVIDENCE] Supabase unavailable. "
                "Using local evidence."
            )
            return None

        safe_store_id = self._safe_component(
            store_id
        )

        safe_filename = self._safe_component(
            filename
        )

        storage_path = (
            f"{safe_store_id}/"
            f"{safe_filename}"
        )

        try:

            with open(
                local_path,
                "rb",
            ) as file:

                file_bytes = file.read()

            sb.storage.from_(
                self.bucket_name
            ).upload(
                storage_path,
                file_bytes,
                {
                    "content-type": content_type,
                    "upsert": "false",
                },
            )

            print(
                "[EVIDENCE] Uploaded:",
                storage_path,
            )

            return storage_path

        except Exception as exc:

            print(
                "[EVIDENCE] Upload failed:",
                exc,
            )

            return None

    # ============================================================
    # DATABASE RECORD
    # ============================================================

    def _insert_evidence_record(
        self,
        owner_id: Optional[str],
        store_id: str,
        visit_id: Optional[str],
        observed_item_id: Optional[str],
        camera_id: Optional[str],
        person_track_id: Optional[str],
        evidence_type: str,
        storage_path: str,
        captured_at: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Insert evidence metadata into:

            public.security_evidence

        owner_id is required by the current Supabase schema.
        """

        sb = self._create_supabase_client()

        if sb is None:
            return False

        if not owner_id:
            print(
                "[EVIDENCE] owner_id is required "
                "for security_evidence."
            )
            return False

        payload = {
            "owner_id": str(owner_id),
            "store_id": str(store_id),
            "evidence_type": str(evidence_type),
            "storage_path": str(storage_path),
            "captured_at": captured_at,
            "metadata": metadata or {},
        }

        if visit_id:
            payload["visit_id"] = str(
                visit_id
            )

        if observed_item_id:
            payload["observed_item_id"] = str(
                observed_item_id
            )

        if camera_id:
            payload["camera_id"] = str(
                camera_id
            )

        if person_track_id:
            payload["person_track_id"] = str(
                person_track_id
            )

        try:

            result = (
                sb.table(
                    "security_evidence"
                )
                .insert(payload)
                .execute()
            )

            if getattr(
                result,
                "data",
                None,
            ):

                print(
                    "[EVIDENCE] Database record created."
                )

                return True

            print(
                "[EVIDENCE] Database insert returned no data."
            )

            return False

        except Exception as exc:

            print(
                "[EVIDENCE] Database insert failed:",
                exc,
            )

            return False

    # ============================================================
    # EVENT FRAME HELPER
    # ============================================================

    def record_event_frame(
        self,
        frame,
        store_id: str,
        owner_id: Optional[str] = None,
        visit_id: Optional[str] = None,
        person_track_id: Optional[str] = None,
        observed_item_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        event_type: str = "security_event",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Convenience method for recording an event frame.
        """

        event_metadata = dict(
            metadata or {}
        )

        event_metadata.setdefault(
            "event_type",
            event_type,
        )

        return self.save_snapshot(
            frame=frame,
            store_id=store_id,
            owner_id=owner_id,
            visit_id=visit_id,
            person_track_id=person_track_id,
            observed_item_id=observed_item_id,
            camera_id=camera_id,
            metadata=event_metadata,
        )

    # ============================================================
    # SIGNED URL
    # ============================================================

    def create_signed_url(
        self,
        storage_path: str,
        expires_in: int = 3600,
    ) -> Optional[str]:
        """
        Create a temporary signed URL for private evidence.
        """

        sb = self._create_supabase_client()

        if sb is None:
            return None

        try:

            result = (
                sb.storage
                .from_(self.bucket_name)
                .create_signed_url(
                    storage_path,
                    expires_in,
                )
            )

            if isinstance(result, dict):

                return (
                    result.get("signedURL")
                    or result.get("signedUrl")
                    or result.get("signed_url")
                )

            return None

        except Exception as exc:

            print(
                "[EVIDENCE] Signed URL error:",
                exc,
            )

            return None


# ================================================================
# SIMPLE LOCAL TEST
# ================================================================

if __name__ == "__main__":

    print(
        "Store Vision AI EvidenceRecorder"
    )

    print(
        "Bucket:",
        "security-evidence",
    )

    print(
        "Local directory:",
        "evidence",
    )

    print(
        "No camera or facial recognition is started "
        "by this test."
    )