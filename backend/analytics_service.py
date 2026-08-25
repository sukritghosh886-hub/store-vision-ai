from collections import Counter
from typing import Dict, List


class VisitorAnalytics:
    """
    Analytics calculated from visitor event records.

    This class does not directly access Supabase.
    Keeping analytics separate makes it easier to test.
    """

    @staticmethod
    def summarize(
        events: List[Dict],
    ) -> Dict:

        entries = 0
        exits = 0

        for event in events:

            event_type = str(
                event.get("event_type", "")
            ).lower()

            if event_type == "entry":
                entries += 1

            elif event_type == "exit":
                exits += 1

        return {
            "total_events": len(events),
            "entries": entries,
            "exits": exits,
            "net_visitor_change": entries - exits,
        }

    @staticmethod
    def by_camera(
        events: List[Dict],
    ) -> Dict[str, Dict]:

        cameras = {}

        for event in events:

            camera_id = (
                event.get("camera_id")
                or "unknown"
            )

            if camera_id not in cameras:
                cameras[camera_id] = {
                    "entries": 0,
                    "exits": 0,
                    "events": 0,
                }

            cameras[camera_id]["events"] += 1

            event_type = str(
                event.get("event_type", "")
            ).lower()

            if event_type == "entry":
                cameras[camera_id]["entries"] += 1

            elif event_type == "exit":
                cameras[camera_id]["exits"] += 1

        return cameras