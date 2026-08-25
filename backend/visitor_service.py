from typing import Dict, List, Optional

from backend.supabase_client import create_supabase_client


class VisitorService:
    """
    Database operations for Store Vision AI visitor events.
    """

    def __init__(self):
        self.client = create_supabase_client()

    def record_event(
        self,
        event: Dict,
    ) -> Dict:

        response = (
            self.client
            .table("visitor_events")
            .insert(event)
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase did not return the inserted visitor event."
            )

        return response.data[0]

    def get_recent_events(
        self,
        store_id: str,
        limit: int = 100,
    ) -> List[Dict]:

        response = (
            self.client
            .table("visitor_events")
            .select("*")
            .eq("store_id", store_id)
            .order(
                "occurred_at",
                desc=True,
            )
            .limit(limit)
            .execute()
        )

        return response.data or []

    def get_events_for_camera(
        self,
        store_id: str,
        camera_id: str,
        limit: int = 100,
    ) -> List[Dict]:

        response = (
            self.client
            .table("visitor_events")
            .select("*")
            .eq("store_id", store_id)
            .eq("camera_id", camera_id)
            .order(
                "occurred_at",
                desc=True,
            )
            .limit(limit)
            .execute()
        )

        return response.data or []