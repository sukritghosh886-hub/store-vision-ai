from typing import Dict, List, Optional

from backend.supabase_client import create_supabase_client


class StoreService:
    """
    Database operations for stores.
    """

    def __init__(self):
        self.client = create_supabase_client()

    def create_store(
        self,
        owner_id: str,
        name: str,
        location: Optional[str] = None,
    ) -> Dict:

        payload = {
            "owner_id": owner_id,
            "name": name,
            "location": location,
        }

        response = (
            self.client
            .table("stores")
            .insert(payload)
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase did not return the created store."
            )

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

    def get_owner_stores(
        self,
        owner_id: str,
    ) -> List[Dict]:

        response = (
            self.client
            .table("stores")
            .select("*")
            .eq("owner_id", owner_id)
            .order(
                "created_at",
                desc=False,
            )
            .execute()
        )

        return response.data or []