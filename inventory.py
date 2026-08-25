from .supabase_client import create_supabase_client


def add_stock(user_id, product_id, quantity, note=""):
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
    new_stock = current_stock + quantity

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

    transaction = {
        "user_id": user_id,
        "product_id": product_id,
        "transaction_type": "stock_in",
        "quantity": quantity,
        "note": note
    }

    return (
        client
        .table("inventory_transactions")
        .insert(transaction)
        .execute()
    ).data


def remove_stock(user_id, product_id, quantity, note=""):
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

    transaction = {
        "user_id": user_id,
        "product_id": product_id,
        "transaction_type": "stock_out",
        "quantity": quantity,
        "note": note
    }

    return (
        client
        .table("inventory_transactions")
        .insert(transaction)
        .execute()
    ).data