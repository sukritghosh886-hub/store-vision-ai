def save_shelf_scan(
    supabase,
    user_id,
    image_path,
    result,
):

    data = {
        "user_id": user_id,
        "image_path": image_path,
        "total_slots":
            result["total_slots"],
        "occupied_slots":
            result["occupied_slots"],
        "empty_slots":
            result["empty_slots"],
        "stock_status":
            result["stock_status"],
    }

    return (
        supabase
        .table("shelf_scans")
        .insert(data)
        .execute()
    )