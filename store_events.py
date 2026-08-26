from datetime import datetime, timezone

from backend.supabase_client import create_supabase_client


def _client():
    return create_supabase_client()


def get_or_create_store(
    name: str,
    address: str = None,
) -> str:
    client = _client()

    existing = (
        client
        .table("stores")
        .select("id")
        .eq("name", name)
        .limit(1)
        .execute()
    )

    if existing.data:
        return existing.data[0]["id"]

    created = (
        client
        .table("stores")
        .insert(
            {
                "name": name,
                "address": address,
            }
        )
        .execute()
    )

    if not created.data:
        raise RuntimeError("Could not create store.")

    return created.data[0]["id"]


def start_visit(
    store_id: str,
    track_id: int,
    camera_id: str = "default",
) -> str:
    client = _client()

    row = {
        "store_id": store_id,
        "track_id": track_id,
        "camera_id": camera_id,
        "status": "in_store",
    }

    result = (
        client
        .table("visits")
        .insert(row)
        .execute()
    )

    if not result.data:
        raise RuntimeError("Could not create visit.")

    return result.data[0]["id"]


def log_item_event(
    visit_id: str,
    item_label: str,
    zone: str,
    confidence: float = None,
):
    client = _client()

    result = (
        client
        .table("item_events")
        .insert(
            {
                "visit_id": visit_id,
                "item_label": item_label,
                "zone": zone,
                "confidence": confidence,
            }
        )
        .execute()
    )

    return result.data


def log_billing_event(
    visit_id: str,
    item_label: str = None,
    quantity: int = 1,
    source: str = "manual",
):
    client = _client()

    result = (
        client
        .table("billing_events")
        .insert(
            {
                "visit_id": visit_id,
                "item_label": item_label,
                "quantity": quantity,
                "source": source,
            }
        )
        .execute()
    )

    return result.data


def get_item_events(visit_id: str):
    client = _client()

    return (
        client
        .table("item_events")
        .select("*")
        .eq("visit_id", visit_id)
        .order("detected_at", desc=True)
        .execute()
        .data
        or []
    )


def get_billing_events(visit_id: str):
    client = _client()

    return (
        client
        .table("billing_events")
        .select("*")
        .eq("visit_id", visit_id)
        .order("billed_at", desc=True)
        .execute()
        .data
        or []
    )


def get_unpaid_count(visit_id: str) -> int:
    client = _client()

    items = (
        client
        .table("item_events")
        .select("item_label")
        .eq("visit_id", visit_id)
        .eq("zone", "shelf")
        .execute()
        .data
        or []
    )

    bills = (
        client
        .table("billing_events")
        .select("item_label,quantity")
        .eq("visit_id", visit_id)
        .execute()
        .data
        or []
    )

    detected = {}

    for item in items:
        label = item.get("item_label", "unknown")
        detected[label] = detected.get(label, 0) + 1

    billed = {}

    for bill in bills:
        label = bill.get("item_label", "unknown")
        quantity = int(bill.get("quantity", 1))
        billed[label] = billed.get(label, 0) + quantity

    unpaid = 0

    for label, count in detected.items():
        unpaid += max(
            count - billed.get(label, 0),
            0,
        )

    return unpaid


def close_visit(
    visit_id: str,
    store_id: str,
):
    client = _client()

    unpaid = get_unpaid_count(visit_id)

    status = (
        "exited_flagged"
        if unpaid > 0
        else "exited_clean"
    )

    client.table("visits").update(
        {
            "exited_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "status": status,
            "unpaid_item_count": unpaid,
        }
    ).eq(
        "id",
        visit_id,
    ).execute()

    if unpaid <= 0:
        return None

    alert = (
        client
        .table("alerts")
        .insert(
            {
                "visit_id": visit_id,
                "store_id": store_id,
                "alert_type": "unpaid_item_flag",
                "unpaid_item_count": unpaid,
                "status": "open",
            }
        )
        .execute()
    )

    return (
        alert.data[0]
        if alert.data
        else None
    )


def get_open_alerts(
    store_id: str = None,
):
    client = _client()

    query = (
        client
        .table("alerts")
        .select("*")
        .eq("status", "open")
        .order("created_at", desc=True)
    )

    if store_id:
        query = query.eq(
            "store_id",
            store_id,
        )

    return query.execute().data or []


def get_all_alerts(
    store_id: str = None,
    limit: int = 100,
):
    client = _client()

    query = (
        client
        .table("alerts")
        .select("*")
        .order(
            "created_at",
            desc=True,
        )
        .limit(limit)
    )

    if store_id:
        query = query.eq(
            "store_id",
            store_id,
        )

    return query.execute().data or []


def review_alert(
    alert_id: str,
    new_status: str,
    reviewed_by: str,
    notes: str = None,
):
    if new_status not in (
        "confirmed",
        "dismissed",
    ):
        raise ValueError(
            "Status must be confirmed or dismissed."
        )

    client = _client()

    return (
        client
        .table("alerts")
        .update(
            {
                "status": new_status,
                "reviewed_by": reviewed_by,
                "reviewed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "notes": notes,
            }
        )
        .eq(
            "id",
            alert_id,
        )
        .execute()
        .data
    )


def get_visit_history(
    store_id: str,
    limit: int = 50,
):
    client = _client()

    return (
        client
        .table("visits")
        .select("*")
        .eq("store_id", store_id)
        .order(
            "entered_at",
            desc=True,
        )
        .limit(limit)
        .execute()
        .data
        or []
    )