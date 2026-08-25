from .supabase_client import create_supabase_client


def get_dashboard_data(user_id):
    client = create_supabase_client()

    products = (
        client
        .table("products")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    ).data or []

    sales = (
        client
        .table("sales")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    ).data or []

    total_products = len(products)

    total_stock = sum(
        p.get("stock_quantity", 0)
        for p in products
    )

    inventory_value = sum(
        float(p.get("cost_price", 0))
        * p.get("stock_quantity", 0)
        for p in products
    )

    total_sales = sum(
        float(s.get("total_amount", 0))
        for s in sales
    )

    total_items_sold = sum(
        s.get("quantity", 0)
        for s in sales
    )

    low_stock = [
        p
        for p in products
        if p.get("stock_quantity", 0)
        <= p.get("minimum_stock", 5)
    ]

    return {
        "total_products": total_products,
        "total_stock": total_stock,
        "inventory_value": inventory_value,
        "total_sales": total_sales,
        "total_items_sold": total_items_sold,
        "low_stock": low_stock,
        "products": products,
        "sales": sales
    }