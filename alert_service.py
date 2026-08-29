def create_empty_shelf_alert(
    supabase,
    user_id,
    empty_slots,
):

    return (
        supabase
        .table("alerts")
        .insert(
            {
                "user_id": user_id,
                "alert_type":
                    "empty_shelf",

                "unpaid_item_count": 0,

                "status": "open",

                "notes":
                    (
                        f"{empty_slots} "
                        "empty shelf positions "
                        "detected."
                    ),
            }
        )
        .execute()
    )