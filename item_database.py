"""
Store Vision AI
Database bridge for detected item events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def create_observed_item(
    sb,
    owner_id: str,
    store_id: str,
    visit_id: str,
    camera_id: str,
    event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    row = {

        "owner_id":
            owner_id,

        "store_id":
            store_id,

        "visit_id":
            visit_id,

        "camera_id":
            camera_id,

        "person_track_id":
            event.get(
                "person_track_id"
            ),

        "detected_label":
            event.get(
                "detected_label"
            ),

        "quantity":
            1,

        "confidence":
            event.get(
                "confidence"
            ),

        "picked_at":
            utc_now(),

        "status":
            "observed",

        "bbox":
            event.get(
                "bbox"
            ),

        "centroid":
            None,

        "metadata":
            {
                "item_tracker_id":
                    event.get(
                        "item_tracker_id"
                    ),

                "event_type":
                    event.get(
                        "event_type"
                    ),
            },
    }

    result = (
        sb
        .table("observed_items")
        .insert(row)
        .execute()
    )

    if not result.data:

        return None

    return result.data[0]


def mark_item_returned(
    sb,
    observed_item_id: str,
):

    result = (
        sb
        .table("observed_items")
        .update({
            "status": "returned",
            "returned_at": utc_now(),
        })
        .eq(
            "id",
            observed_item_id
        )
        .execute()
    )

    return result.data


__all__ = [
    "create_observed_item",
    "mark_item_returned",
]