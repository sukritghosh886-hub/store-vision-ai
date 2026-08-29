import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from backend.supabase_client import create_supabase_client
from vision_pipeline import process_image
from sku_recognition import recognize_products

st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
)

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #15171b;
    }

    .metric-card {
        background: #1c1f25;
        border: 1px solid #33383f;
        border-radius: 10px;
        padding: 20px;
    }

    .alert-card {
        background: #2a1c16;
        border: 1px solid #8a4920;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .success-card {
        background: #17251b;
        border: 1px solid #3b7848;
        border-radius: 10px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SUPABASE
# ---------------------------------------------------------

@st.cache_resource
def get_supabase():
    return create_supabase_client()


try:
    supabase = get_supabase()
except Exception as exc:
    st.error("Supabase connection failed.")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("👁️ Store Vision AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Vision Scanner",
        "Live Monitor",
        "SKU Recognition",
        "Security Alerts",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Retail Computer Vision Platform")
st.sidebar.caption("CPU-first • Supabase • YOLO • OpenCV")


# ---------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------

def safe_count(table_name):
    try:
        result = (
            supabase
            .table(table_name)
            .select("*", count="exact")
            .limit(1)
            .execute()
        )
        return result.count or 0
    except Exception:
        return 0


def get_products():
    try:
        return (
            supabase
            .table("products")
            .select("*")
            .order("name")
            .execute()
            .data
            or []
        )
    except Exception:
        return []


def get_alerts():
    try:
        return (
            supabase
            .table("alerts")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
            .data
            or []
        )
    except Exception:
        return []


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("Store Vision AI")
    st.caption(
        "Computer vision for retail inventory, customer visits and security."
    )

    products = get_products()
    alerts = get_alerts()

    open_alerts = [
        a for a in alerts
        if a.get("status") == "open"
    ]

    visits = safe_count("visits")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Products",
            len(products),
        )

    with col2:
        st.metric(
            "Tracked Visits",
            visits,
        )

    with col3:
        st.metric(
            "Open Alerts",
            len(open_alerts),
        )

    with col4:
        low_stock = sum(
            1
            for p in products
            if int(p.get("stock_quantity", 0))
            <= int(p.get("reorder_level", 5))
        )

        st.metric(
            "Low Stock",
            low_stock,
        )

    st.markdown("## System status")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("✓ Supabase connected")

    with c2:
        st.success("✓ Vision engine available")

    with c3:
        if open_alerts:
            st.warning(
                f"{len(open_alerts)} alert(s) require review"
            )
        else:
            st.success("✓ No open security alerts")

    st.markdown("## Inventory")

    if products:

        rows = []

        for product in products:
            stock = int(
                product.get(
                    "stock_quantity",
                    0,
                )
            )

            reorder = int(
                product.get(
                    "reorder_level",
                    5,
                )
            )

            rows.append(
                {
                    "SKU": product.get(
                        "sku",
                        "",
                    ),
                    "Product": product.get(
                        "name",
                        "",
                    ),
                    "Stock": stock,
                    "Reorder Level": reorder,
                    "Status": (
                        "LOW STOCK"
                        if stock <= reorder
                        else "OK"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No products registered yet. "
            "Add products before using SKU recognition."
        )


# =========================================================
# VISION SCANNER
# =========================================================

elif page == "Vision Scanner":

    st.title("🔎 Vision Scanner")

    st.write(
        "Upload a shelf image. YOLO detects people and supported objects, "
        "then the SKU layer attempts to map detections to registered products."
    )

    uploaded = st.file_uploader(
        "Upload image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    confidence = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        0.40,
        0.05,
    )

    if uploaded:

        image_bytes = uploaded.read()

        if st.button(
            "Run Vision Analysis",
            type="primary",
        ):

            with st.spinner(
                "Running YOLO + OpenCV..."
            ):

                result = process_image(
                    image_bytes,
                    confidence=confidence,
                )

            st.image(
                result["image"],
                caption="Detection result",
                use_container_width=True,
            )

            st.success(
                f"{len(result['detections'])} detection(s)"
            )

            if result["detections"]:

                df = pd.DataFrame(
                    result["detections"]
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# LIVE MONITOR
# =========================================================

elif page == "Live Monitor":

    st.title("📹 Live Monitor")

    st.warning(
        "Browser-based webcam streaming inside Streamlit is "
        "different from processing an uploaded video. "
        "For reliable cloud deployment, use an uploaded video "
        "or a camera/RTSP worker."
    )

    video = st.file_uploader(
        "Upload store video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
        ],
    )

    if video:

        if st.button(
            "Start Video Analysis",
            type="primary",
        ):

            suffix = Path(
                video.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(
                    video.read()
                )

                video_path = tmp.name

            st.info(
                "Processing video..."
            )

            cap = cv2.VideoCapture(
                video_path
            )

            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            frame_number = 0

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                frame_number += 1

                if frame_number % 3 != 0:
                    continue

                result = process_image(
                    frame,
                    confidence=0.40,
                    input_is_frame=True,
                )

                frame_placeholder.image(
                    result["image"],
                    channels="BGR",
                    use_container_width=True,
                )

                status_placeholder.write(
                    f"Processed frame {frame_number}"
                )

            cap.release()

            try:
                os.unlink(video_path)
            except OSError:
                pass

            st.success(
                "Video analysis complete."
            )


# =========================================================
# SKU RECOGNITION
# =========================================================

elif page == "SKU Recognition":

    st.title("🏷️ SKU Recognition")

    st.write(
        "Register a product reference image and use it as the "
        "visual template for SKU matching."
    )

    products = get_products()

    if not products:
        st.warning(
            "No products found in Supabase."
        )
    else:

        product_names = {
            p.get("name", "Unknown"): p
            for p in products
        }

        selected_name = st.selectbox(
            "Product",
            list(product_names.keys()),
        )

        selected_product = product_names[
            selected_name
        ]

        reference = st.file_uploader(
            "Reference image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key="sku_reference",
        )

        if reference:

            st.image(
                reference,
                caption=selected_name,
                width=300,
            )

            if st.button(
                "Save SKU Reference",
                type="primary",
            ):

                try:

                    recognize_products.save_reference(
                        supabase,
                        selected_product["id"],
                        reference.read(),
                    )

                    st.success(
                        "SKU reference saved."
                    )

                except Exception as exc:
                    st.error(
                        "Could not save SKU reference."
                    )
                    st.exception(exc)


# =========================================================
# SECURITY ALERTS
# =========================================================

elif page == "Security Alerts":

    st.title("🚨 Security Alerts")

    alerts = get_alerts()

    open_alerts = [
        a
        for a in alerts
        if a.get("status") == "open"
    ]

    if not open_alerts:

        st.success(
            "No open alerts."
        )

    else:

        st.warning(
            f"{len(open_alerts)} alert(s) require human review."
        )

        for alert in open_alerts:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Potential billing discrepancy"
                )

                st.write(
                    f"Visit: {alert.get('visit_id', '-')}"
                )

                st.write(
                    "Unpaid items:",
                    alert.get(
                        "unpaid_item_count",
                        0,
                    ),
                )

                c1, c2 = st.columns(2)

                with c1:

                    if st.button(
                        "Confirm",
                        key=f"confirm_{alert['id']}",
                    ):

                        supabase.table(
                            "alerts"
                        ).update(
                            {
                                "status": "confirmed",
                            }
                        ).eq(
                            "id",
                            alert["id"],
                        ).execute()

                        st.rerun()

                with c2:

                    if st.button(
                        "Dismiss",
                        key=f"dismiss_{alert['id']}",
                    ):

                        supabase.table(
                            "alerts"
                        ).update(
                            {
                                "status": "dismissed",
                            }
                        ).eq(
                            "id",
                            alert["id"],
                        ).execute()

                        st.rerun()