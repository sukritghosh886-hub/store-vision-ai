from .supabase_client import create_supabase_client


def get_products(user_id):
    client = create_supabase_client()

    response = (
        client
        .table("products")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


def create_product(
    user_id,
    name,
    sku,
    category,
    cost_price,
    selling_price,
    stock_quantity,
    minimum_stock,
    supplier
):
    client = create_supabase_client()

    data = {
        "user_id": user_id,
        "name": name,
        "sku": sku,
        "category": category,
        "cost_price": cost_price,
        "selling_price": selling_price,
        "stock_quantity": stock_quantity,
        "minimum_stock": minimum_stock,
        "supplier": supplier
    }

    response = (
        client
        .table("products")
        .insert(data)
        .execute()
    )

    return response.data


def delete_product(product_id, user_id):
    client = create_supabase_client()

    response = (
        client
        .table("products")
        .delete()
        .eq("id", product_id)
        .eq("user_id", user_id)
        .execute()
    )

    return response.data