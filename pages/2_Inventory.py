import streamlit as st

from backend.database import get_table


st.set_page_config(
    page_title="Inventory",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Inventory Management")

products = get_table("products")

if not products:

    st.info(
        "No products available."
    )

else:

    total_value = sum(
        float(
            p.get("stock_quantity") or 0
        )
        * float(
            p.get("cost_price") or 0
        )
        for p in products
    )

    retail_value = sum(
        float(
            p.get("stock_quantity") or 0
        )
        * float(
            p.get("selling_price") or 0
        )
        for p in products
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Inventory Cost Value",
        f"₹{total_value:,.2f}",
    )

    c2.metric(
        "Inventory Retail Value",
        f"₹{retail_value:,.2f}",
    )

    rows = []

    for p in products:

        stock = int(
            p.get("stock_quantity") or 0
        )

        minimum = int(
            p.get("minimum_stock") or 0
        )

        rows.append(
            {
                "Name": p.get("name"),
                "SKU": p.get("sku"),
                "Category": p.get("category"),
                "Stock": stock,
                "Minimum": minimum,
                "Cost": p.get("cost_price"),
                "Selling Price": p.get(
                    "selling_price"
                ),
                "Status": (
                    "LOW STOCK"
                    if stock <= minimum
                    else "OK"
                ),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )