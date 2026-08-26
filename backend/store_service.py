from typing import Dict, List, Optional

from backend.supabase_client import create_supabase_client


class StoreService:
    """Database operations for stores."""

    def __init__(self):
        self.client = create_supabase_client()

    def create_store(
        self,
        name: str,
        address: Optional[str] = None,
    ) -> Dict:
        payload = {
            "name": name,
            "address": address,
        }

        response = (
            self.client
            .table("stores")
            .insert(payload)
            .execute()
        )

        if not response.data:
            raise RuntimeError("Store was not created.")

        return response.data[0]

    def get_store(
        self,
        store_id: str,
    ) -> Optional[Dict]:
        response = (
            self.client
            .table("stores")
            .select("*")
            .eq("id", store_id)
            .maybe_single()
            .execute()
        )

        return response.data

    def get_all_stores(self) -> List[Dict]:
        response = (
            self.client
            .table("stores")
            .select("*")
            .order("created_at", desc=False)
            .execute()
        )

        return response.data or []

    def get_or_create_store(
        self,
        name: str,
        address: Optional[str] = None,
    ) -> Dict:
        response = (
            self.client
            .table("stores")
            .select("*")
            .eq("name", name)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

        return self.create_store(
            name=name,
            address=address,
        )