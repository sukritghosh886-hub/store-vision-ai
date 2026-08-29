from datetime import datetime, timezone


def save_shelf_scan(
    supabase,
    user_id,
    image_path,
    summary,
):

    payload = {

        "user_id":
            user_id,

        "image_path":
            image_path,

        "total_slots":
            summary["total_zones"],

        "occupied_slots":
            (
                summary["total_zones"]
                -
                summary["empty_zones"]
            ),

        "empty_slots":
            summary["empty_zones"],

        "stock_status":
            (
                "critical"
                if summary["empty_zones"] > 3
                else
                "attention"
                if summary["empty_zones"] > 0
                else
                "healthy"
            ),

    }

    return (
        supabase
        .table("shelf_scans")
        .insert(payload)
        .execute()
    )


def create_shelf_alert(
    supabase,
    user_id,
    empty_slots,
):

    if empty_slots <= 0:

        return None

    payload = {

        "user_id":
            user_id,

        "alert_type":
            "empty_shelf",

        "unpaid_item_count":
            0,

        "status":
            "open",

        "notes":
            (
                f"Detected "
                f"{empty_slots} "
                "empty shelf zone(s). "
                f"Created at "
                f"{datetime.now(timezone.utc).isoformat()}"
            ),
    }

    return (
        supabase
        .table("alerts")
        .insert(payload)
        .execute()
    )