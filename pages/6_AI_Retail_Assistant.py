import streamlit as st
import pandas as pd

from backend.database import get_table


st.set_page_config(
    page_title="AI Retail Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Retail Assistant")

st.write(
    "Store intelligence generated from your operational data."
)

products = get_table("products")
sales = get_table("sales")
visits = get_table("store_visits")

if not visits:
    visits = get_table("visits")

st.header("📊 Store Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Products",
    len(products),
)

col2.metric(
    "Sales",
    len(sales),
)

col3.metric(
    "Visits",
    len(visits),
)

st.divider()

st.header("🧠 Automatic Insights")

if products:

    low_stock = [
        p
        for p in products
        if int(
            p.get("stock_quantity") or 0
        )
        <= int(
            p.get("minimum_stock") or 0
        )
    ]

    if low_stock:

        st.warning(
            f"{len(low_stock)} product(s) "
            "need inventory attention."
        )

        for product in low_stock:

            st.write(
                f"• **{product.get('name')}** — "
                f"stock: "
                f"{product.get('stock_quantity')}"
            )

    else:

        st.success(
            "Inventory levels are currently healthy."
        )

if sales:

    df = pd.DataFrame(sales)

    if "total_amount" in df.columns:

        df["total_amount"] = pd.to_numeric(
            df["total_amount"],
            errors="coerce",
        ).fillna(0)

        revenue = df[
            "total_amount"
        ].sum()

        st.info(
            f"Recorded revenue: "
            f"₹{revenue:,.2f}"
        )

if visits:

    st.success(
        f"The system has recorded "
        f"{len(visits)} visitor records."
    )

st.divider()

st.caption(
    "Future version: connect this layer to an LLM "
    "for natural-language retail recommendations."
)