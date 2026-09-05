import streamlit as st
import pandas as pd

from backend.database import get_table


st.set_page_config(
    page_title="Visitors",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Visitor Analytics")

visits = get_table(
    "store_visits"
)

if not visits:

    visits = get_table(
        "visits"
    )

if not visits:

    st.info(
        "No visitor data available."
    )

else:

    df = pd.DataFrame(visits)

    st.metric(
        "Total Visits",
        len(df),
    )

    if "entered_at" in df.columns:

        df["entered_at"] = pd.to_datetime(
            df["entered_at"],
            errors="coerce",
        )

        daily = (
            df.dropna(
                subset=["entered_at"]
            )
            .groupby(
                df["entered_at"].dt.date
            )
            .size()
        )

        st.subheader(
            "Visitors per Day"
        )

        st.bar_chart(daily)

    st.subheader(
        "Visit Records"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )