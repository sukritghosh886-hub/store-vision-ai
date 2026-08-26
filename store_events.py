"""
Data-access layer for Store Vision AI.

Wraps Supabase table operations for visits, item events, billing events,
and review alerts. Import what you need from your Streamlit pages:

    from store_events import start_visit, log_item_event, close_visit

Kept import-safe for both "python -m" package execution and Streamlit's
per-page script execution, since those resolve modules differently.
"""
from datetime import datetime, timezone

try:
    from supabase_client import create_supabase_client
except ImportError:
    from .supabase_client import create_supabase_client


def _client():
    return create_supabase_client()


def get_or_create_store(name: str, address: str = None) -> str:
    """Return the store id for `name`, creating it if it doesn't exist yet."""
    client = _client()
    existing = client.table("stores").select("id").eq("name", name).execute()
    if existing.data:
        return existing.data[0]["id"]
    created = client.table("stores").insert({"name": name, "address": address}).execute()
    return created.data[0]["id"]


def start_visit(store_id: str, track_id: int, camera_id: str = "default") -> str:
    """Create a new visit row the first time a tracked person is seen. Returns the visit id."""
    client = _client()
    row = {
        "store_id": store_id,
        "track_id": track_id,
        "camera_id": camera_id,
        "status": "in_store",
    }
    result = client.table("visits").insert(row).execute()
    return result.data[0]["id"]


def log_item_event(visit_id: str, item_label: str, zone: str, confidence: float = None):
    """Record that an item-like object was seen near this tracked person."""
    client = _client()
    client.table("item_events").insert({
        "visit_id": visit_id,
        "item_label": item_label,
        "zone": zone,
        "confidence": confidence,
    }).execute()


def log_billing_event(visit_id: str, item_label: str = None, quantity: int = 1, source: str = "manual"):
    """Record that an item was paid for — via the demo checkout button or a simulated POS feed."""
    client = _client()
    client.table("billing_events").insert({
        "visit_id": visit_id,
        "item_label": item_label,
        "quantity": quantity,
        "source": source,
    }).execute()


def get_unpaid_count(visit_id: str) -> int:
    """Items picked up near the shelf zone minus items billed, for this visit. Never negative."""
    client = _client()
    picked = (
        client.table("item_events")
        .select("id", count="exact")
        .eq("visit_id", visit_id)
        .eq("zone", "shelf")
        .execute()
    )
    billed = client.table("billing_events").select("quantity").eq("visit_id", visit_id).execute()
    picked_count = picked.count or 0
    billed_count = sum(row["quantity"] for row in billed.data) if billed.data else 0
    return max(picked_count - billed_count, 0)


def close_visit(visit_id: str, store_id: str) -> dict:
    """
    Call when a tracked person crosses the exit zone (or their track is lost).
    Closes the visit and raises an alert if items were never billed.
    Returns the alert row if one was raised, else None.
    """
    client = _client()
    unpaid = get_unpaid_count(visit_id)
    status = "exited_flagged" if unpaid > 0 else "exited_clean"

    client.table("visits").update({
        "exited_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "unpaid_item_count": unpaid,
    }).eq("id", visit_id).execute()

    if unpaid > 0:
        alert = client.table("alerts").insert({
            "visit_id": visit_id,
            "store_id": store_id,
            "alert_type": "unpaid_item_flag",
            "unpaid_item_count": unpaid,
            "status": "open",
        }).execute()
        return alert.data[0]
    return None


def get_open_alerts(store_id: str = None) -> list:
    client = _client()
    query = client.table("alerts").select("*").eq("status", "open").order("created_at", desc=True)
    if store_id:
        query = query.eq("store_id", store_id)
    return query.execute().data


def review_alert(alert_id: str, new_status: str, reviewed_by: str, notes: str = None):
    """new_status: 'confirmed' (genuinely unpaid) or 'dismissed' (false positive)."""
    client = _client()
    client.table("alerts").update({
        "status": new_status,
        "reviewed_by": reviewed_by,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }).eq("id", alert_id).execute()


def get_visit_history(store_id: str, limit: int = 50) -> list:
    client = _client()
    return (
        client.table("visits")
        .select("*")
        .eq("store_id", store_id)
        .order("entered_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )


def get_all_alerts(store_id: str = None, limit: int = 100) -> list:
    client = _client()
    query = client.table("alerts").select("*").order("created_at", desc=True).limit(limit)
    if store_id:
        query = query.eq("store_id", store_id)
    return query.execute().data
