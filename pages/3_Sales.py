import streamlit as st
import pandas as pd

from backend.database import get_table


st.set_page_config(
    page_title="Sales",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Sales Analytics")

sales = get_table("sales")

if not sales:

    st.info(
        "No sales recorded yet."
    )

else:

    df = pd.DataFrame(sales)

    df["total_amount"] = pd.to_numeric(
        df["total_amount"],
        errors="coerce",
    ).fillna(0)

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce",
    ).fillna(0)

    revenue = df[
        "total_amount"
    ].sum()

    units = df[
        "quantity"
    ].sum()

    transactions = len(df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Revenue",
        f"₹{revenue:,.2f}",
    )

    c2.metric(
        "Units Sold",
        int(units),
    )

    c3.metric(
        "Transactions",
        transactions,
    )

    st.subheader(
        "Sales Transactions"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    if "created_at" in df.columns:

        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce",
        )

        daily = (
            df.dropna(
                subset=["created_at"]
            )
            .groupby(
                df["created_at"].dt.date
            )["total_amount"]
            .sum()
        )

        st.subheader(
            "Daily Revenue"
        )

        st.line_chart(daily)