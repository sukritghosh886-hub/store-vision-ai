from __future__ import annotations

from datetime import datetime, timezone

from pathlib import Path

import cv2


class EvidenceRecorder:

    def __init__(
        self,
        directory: str = "evidence"
    ):

        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_frame(
        self,
        frame,
        track_id: int,
        item_name: str
    ) -> str:

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        safe_item = "".join(
            character
            if (
                character.isalnum()
                or character in "-_"
            )
            else "_"

            for character
            in item_name
        )

        filename = (
            f"suspected_theft_"
            f"person_{track_id}_"
            f"{safe_item}_"
            f"{timestamp}.jpg"
        )

        path = (
            self.directory /
            filename
        )

        success = cv2.imwrite(
            str(path),
            frame
        )

        if not success:

            raise IOError(
                f"Unable to save evidence: {path}"
            )

        return str(path)