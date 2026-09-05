from __future__ import annotations

import streamlit as st

from backend.database import get_table


st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
)


def safe_int(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=30)
def load_products():
    return get_table("products")


@st.cache_data(ttl=30)
def load_sales():
    return get_table("sales")


@st.cache_data(ttl=30)
def load_alerts():
    return get_table("security_alerts")


@st.cache_data(ttl=30)
def load_visits():
    return get_table("visits")


def main():

    st.title("👁️ Store Vision AI")
    st.caption(
        "Computer Vision × Retail Analytics × Security Intelligence"
    )

    st.sidebar.title("Store Vision AI")

    products = load_products()
    sales = load_sales()
    alerts = load_alerts()
    visits = load_visits()

    total_products = len(products)

    units_sold = sum(
        safe_int(
            row.get("quantity")
        )
        for row in sales
    )

    revenue = sum(
        safe_float(
            row.get("total_amount")
        )
        for row in sales
    )

    low_stock = sum(
        1
        for product in products
        if safe_int(
            product.get("stock_quantity")
        )
        <= safe_int(
            product.get("minimum_stock")
        )
    )

    open_alerts = sum(
        1
        for alert in alerts
        if str(
            alert.get("status", "")
        ).lower()
        in {
            "open",
            "pending",
            "unresolved",
        }
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Products",
        total_products,
    )

    c2.metric(
        "Units Sold",
        units_sold,
    )

    c3.metric(
        "Revenue",
        f"₹{revenue:,.2f}",
    )

    c4.metric(
        "Low Stock",
        low_stock,
    )

    c5.metric(
        "Open Alerts",
        open_alerts,
    )

    st.divider()

    st.header("📦 Inventory")

    if products:

        rows = []

        for product in products:

            stock = safe_int(
                product.get("stock_quantity")
            )

            minimum = safe_int(
                product.get("minimum_stock")
            )

            rows.append(
                {
                    "Product": product.get(
                        "name",
                        "Unknown",
                    ),
                    "SKU": product.get(
                        "sku",
                        "",
                    ),
                    "Category": product.get(
                        "category",
                        "",
                    ),
                    "Stock": stock,
                    "Minimum": minimum,
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

    else:
        st.info(
            "No products found."
        )

    st.divider()

    st.header("🚨 Recent Security Alerts")

    if alerts:

        recent = sorted(
            alerts,
            key=lambda x: str(
                x.get(
                    "created_at",
                    "",
                )
            ),
            reverse=True,
        )[:10]

        st.dataframe(
            [
                {
                    "Type": a.get(
                        "alert_type",
                        "",
                    ),
                    "Severity": a.get(
                        "severity",
                        "",
                    ),
                    "Message": a.get(
                        "message",
                        "",
                    ),
                    "Status": a.get(
                        "status",
                        "",
                    ),
                    "Created": a.get(
                        "created_at",
                        "",
                    ),
                }
                for a in recent
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.success(
            "No security alerts."
        )


if __name__ == "__main__":
    main()