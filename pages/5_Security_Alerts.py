import streamlit as st
import pandas as pd

from backend.database import get_table


st.set_page_config(
    page_title="Security Alerts",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Security Alerts")

alerts = get_table(
    "security_alerts"
)

if not alerts:

    st.success(
        "No security alerts found."
    )

else:

    df = pd.DataFrame(alerts)

    status_options = [
        "ALL",
        "OPEN",
        "RESOLVED",
    ]

    selected = st.selectbox(
        "Filter by status",
        status_options,
    )

    if selected != "ALL":

        df = df[
            df["status"]
            .fillna("")
            .str.upper()
            == selected
        ]

    st.metric(
        "Displayed Alerts",
        len(df),
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )