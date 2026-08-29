from __future__ import annotations

import streamlit as st

from backend.supabase_client import create_supabase_client
from backend.retail_assistant import RetailAssistant


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

@st.cache_resource
def get_supabase():
    return create_supabase_client()


try:
    supabase = get_supabase()
except Exception as exc:
    st.error("Could not connect to Supabase.")
    st.exception(exc)
    st.stop()


# =========================================================
# HELPERS
# =========================================================

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


def get_table(table_name: str, columns="*"):
    try:
        result = (
            supabase
            .table(table_name)
            .select(columns)
            .execute()
        )
        return result.data or []
    except Exception:
        return []


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("👁️ STORE VISION AI")
st.sidebar.caption("Retail Computer Vision Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Vision Scanner",
        "Live Monitor",
        "SKU Recognition",
        "Security Alerts",
        "AI Retail Assistant",
    ],
)

st.sidebar.divider()

st.sidebar.caption("Portfolio Project")
st.sidebar.caption("Computer Vision × Retail Analytics")
st.sidebar.caption("Supabase × Streamlit × YOLO × OpenCV")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Store Vision AI")
    st.subheader("Retail Intelligence Dashboard")

    st.write(
        "Monitor inventory, sales, shelf conditions, visitors and security events "
        "from one unified retail intelligence platform."
    )

    products = get_table("products")
    sales = get_table("sales")
    alerts = get_table("alerts")
    visits = get_table("visits")
    shelf_scans = get_table("shelf_scans")

    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    total_products = len(products)

    total_units_sold = sum(
        safe_int(row.get("quantity"))
        for row in sales
    )

    total_revenue = sum(
        safe_float(row.get("total_amount"))
        for row in sales
    )

    low_stock_count = sum(
        1
        for product in products
        if safe_int(product.get("stock_quantity"))
        <= safe_int(product.get("minimum_stock"))
    )

    open_alert_count = sum(
        1
        for alert in alerts
        if str(alert.get("status", "")).lower() in
        ("open", "pending", "unresolved")
    )

    visitor_count = len(visits)

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric(
        "Products",
        total_products,
    )

    c2.metric(
        "Units Sold",
        total_units_sold,
    )

    c3.metric(
        "Revenue",
        f"₹{total_revenue:,.2f}",
    )

    c4.metric(
        "Low Stock",
        low_stock_count,
    )

    c5.metric(
        "Open Alerts",
        open_alert_count,
    )

    c6.metric(
        "Tracked Visits",
        visitor_count,
    )

    st.divider()

    # -----------------------------------------------------
    # INVENTORY
    # -----------------------------------------------------

    st.header("📦 Inventory Status")

    if products:

        inventory_rows = []

        for product in products:

            stock = safe_int(
                product.get("stock_quantity")
            )

            minimum_stock = safe_int(
                product.get("minimum_stock")
            )

            status = (
                "LOW STOCK"
                if stock <= minimum_stock
                else "OK"
            )

            inventory_rows.append(
                {
                    "Product": product.get(
                        "name",
                        "Unknown"
                    ),
                    "SKU": product.get(
                        "sku",
                        "N/A"
                    ),
                    "Category": product.get(
                        "category",
                        "N/A"
                    ),
                    "Stock": stock,
                    "Minimum Stock": minimum_stock,
                    "Status": status,
                }
            )

        st.dataframe(
            inventory_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No products found in the Supabase products table."
        )

    # -----------------------------------------------------
    # LOW STOCK PRODUCTS
    # -----------------------------------------------------

    if products:

        low_products = [
            product
            for product in products
            if safe_int(product.get("stock_quantity"))
            <= safe_int(product.get("minimum_stock"))
        ]

        if low_products:

            st.subheader("⚠️ Reorder Required")

            for product in low_products:

                stock = safe_int(
                    product.get("stock_quantity")
                )

                minimum = safe_int(
                    product.get("minimum_stock")
                )

                st.warning(
                    f"**{product.get('name', 'Unknown')}** — "
                    f"Stock: {stock} | "
                    f"Minimum: {minimum}"
                )

    # -----------------------------------------------------
    # RECENT SECURITY ALERTS
    # -----------------------------------------------------

    st.subheader("🚨 Recent Security Alerts")

    if alerts:

        recent_alerts = sorted(
            alerts,
            key=lambda x: str(
                x.get("created_at", "")
            ),
            reverse=True,
        )[:5]

        for alert in recent_alerts:

            alert_type = alert.get(
                "alert_type",
                "Unknown alert",
            )

            status = str(
                alert.get(
                    "status",
                    "unknown",
                )
            ).upper()

            unpaid = safe_int(
                alert.get(
                    "unpaid_item_count"
                )
            )

            with st.container(border=True):

                st.markdown(
                    f"**{alert_type}** — `{status}`"
                )

                st.write(
                    f"Unpaid items: {unpaid}"
                )

                if alert.get("notes"):
                    st.caption(
                        alert["notes"]
                    )

    else:

        st.success(
            "No security alerts recorded."
        )

    # -----------------------------------------------------
    # SHELF SCAN SUMMARY
    # -----------------------------------------------------

    st.subheader("🛒 Shelf Scan Summary")

    if shelf_scans:

        total_scans = len(shelf_scans)

        total_slots = sum(
            safe_int(
                scan.get("total_slots")
            )
            for scan in shelf_scans
        )

        occupied_slots = sum(
            safe_int(
                scan.get("occupied_slots")
            )
            for scan in shelf_scans
        )

        empty_slots = sum(
            safe_int(
                scan.get("empty_slots")
            )
            for scan in shelf_scans
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Shelf Scans",
            total_scans,
        )

        s2.metric(
            "Total Slots",
            total_slots,
        )

        s3.metric(
            "Occupied",
            occupied_slots,
        )

        s4.metric(
            "Empty Slots",
            empty_slots,
        )

    else:

        st.info(
            "No shelf scans recorded yet."
        )


# =========================================================
# VISION SCANNER
# =========================================================

elif page == "Vision Scanner":

    st.title("🔍 Vision Scanner")

    st.write(
        "Upload a shelf image for computer-vision analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload shelf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            caption="Uploaded Shelf Image",
            use_container_width=True,
        )

        st.success(
            "Image successfully uploaded."
        )

        st.info(
            "Connect this interface to the existing "
            "`vision_pipeline.py` inference function "
            "for YOLO/OpenCV processing."
        )


# =========================================================
# LIVE MONITOR
# =========================================================

elif page == "Live Monitor":

    st.title("📹 Live Monitor")

    st.write(
        "Real-time retail monitoring interface."
    )

    st.info(
        "The repository contains the live camera/video "
        "pipeline. This page is the dashboard entry point "
        "for continuous monitoring."
    )

    st.markdown(
        """
### Monitoring Pipeline

```text
Camera / Video
      ↓
OpenCV
      ↓
YOLO Detection
      ↓
Object Tracking
      ↓
Visit / Item Events
      ↓
Supabase
      ↓
Dashboard + Alerts