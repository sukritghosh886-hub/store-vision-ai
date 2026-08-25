from .supabase_client import create_supabase_client


def create_sale(
    user_id,
    product_id,
    quantity,
    unit_price,
    customer_name,
    payment_method
):
    client = create_supabase_client()

    product = (
        client
        .table("products")
        .select("stock_quantity")
        .eq("id", product_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not product.data:
        raise ValueError("Product not found.")

    current_stock = product.data["stock_quantity"]

    if quantity > current_stock:
        raise ValueError("Not enough stock.")

    new_stock = current_stock - quantity

    (
        client
        .table("products")
        .update({
            "stock_quantity": new_stock
        })
        .eq("id", product_id)
        .eq("user_id", user_id)
        .execute()
    )

    total_amount = quantity * unit_price

    sale = {
        "user_id": user_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_amount": total_amount,
        "customer_name": customer_name,
        "payment_method": payment_method
    }

    return (
        client
        .table("sales")
        .insert(sale)
        .execute()
    ).data


def get_sales(user_id):
    client = create_supabase_client()

    response = (
        client
        .table("sales")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []