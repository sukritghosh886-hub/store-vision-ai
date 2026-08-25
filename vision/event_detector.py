from datetime import datetime, timezone
from typing import Dict, Optional

from models.person_tracker import PersonTrack
from vision.line_counter import CrossingEvent


class VisitorEventBuilder:
    """
    Converts computer-vision crossing events into records
    suitable for the Supabase visitor_events table.
    """

    def __init__(
        self,
        owner_id: str,
        store_id: str,
        camera_id: str = "camera-1",
    ):
        self.owner_id = owner_id
        self.store_id = store_id
        self.camera_id = camera_id

    def build(
        self,
        crossing: CrossingEvent,
        track: Optional[PersonTrack] = None,
    ) -> Dict:

        metadata = {
            "source": "store_vision_ai",
            "tracking_method": "centroid",
            "point": {
                "x": crossing.point[0],
                "y": crossing.point[1],
            },
        }

        if track is not None:
            metadata["confidence"] = float(
                track.confidence
            )

            metadata["bbox"] = {
                "x": track.bbox[0],
                "y": track.bbox[1],
                "width": track.bbox[2],
                "height": track.bbox[3],
            }

        return {
            "owner_id": self.owner_id,
            "store_id": self.store_id,
            "event_type": crossing.direction,
            "camera_id": self.camera_id,
            "person_track_id": str(
                crossing.track_id
            ),
            "occurred_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "metadata": metadata,
        }