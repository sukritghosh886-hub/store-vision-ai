from typing import Dict, List, Optional

from backend.supabase_client import create_supabase_client


class VisitorService:
    """Database operations for Store Vision AI visits."""

    def __init__(self):
        self.client = create_supabase_client()

    def create_visit(
        self,
        store_id: str,
        track_id: int,
        camera_id: str = "default",
    ) -> Dict:
        payload = {
            "store_id": store_id,
            "track_id": track_id,
            "camera_id": camera_id,
            "status": "in_store",
        }

        response = (
            self.client
            .table("visits")
            .insert(payload)
            .execute()
        )

        if not response.data:
            raise RuntimeError("Visit was not created.")

        return response.data[0]

    def get_visit(self, visit_id: str) -> Optional[Dict]:
        response = (
            self.client
            .table("visits")
            .select("*")
            .eq("id", visit_id)
            .maybe_single()
            .execute()
        )

        return response.data

    def get_recent_visits(
        self,
        store_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        response = (
            self.client
            .table("visits")
            .select("*")
            .eq("store_id", store_id)
            .order("entered_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data or []

    def get_active_visits(
        self,
        store_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        response = (
            self.client
            .table("visits")
            .select("*")
            .eq("store_id", store_id)
            .eq("status", "in_store")
            .order("entered_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data or []

    def mark_exit(
        self,
        visit_id: str,
        flagged: bool = False,
        unpaid_item_count: int = 0,
    ) -> Dict:
        from datetime import datetime, timezone

        payload = {
            "status": (
                "exited_flagged"
                if flagged
                else "exited_clean"
            ),
            "exited_at": datetime.now(timezone.utc).isoformat(),
            "unpaid_item_count": unpaid_item_count,
        }

        response = (
            self.client
            .table("visits")
            .update(payload)
            .eq("id", visit_id)
            .execute()
        )

        if not response.data:
            raise RuntimeError("Visit was not updated.")

        return response.data[0]