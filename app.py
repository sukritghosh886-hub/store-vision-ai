import streamlit as st
import pandas as pd

import store_events


st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
)


st.title("👁️ Store Vision AI")

st.caption(
    "AI-powered retail visitor monitoring, "
    "item observation, billing comparison, "
    "and security alerts."
)


with st.sidebar:

    st.header("Store")

    store_name = st.text_input(
        "Store name",
        "Demo Store",
    )


try:

    store_id = (
        store_events
        .get_or_create_store(
            store_name
        )
    )

except Exception as error:

    st.error(
        "Supabase is not connected."
    )

    st.info(
        "Configure SUPABASE_URL and "
        "SUPABASE_PUBLISHABLE_KEY "
        "in Streamlit Secrets."
    )

    st.exception(error)

    st.stop()


visits = (
    store_events
    .get_visit_history(
        store_id,
        limit=100,
    )
)

alerts = (
    store_events
    .get_all_alerts(
        store_id,
        limit=100,
    )
)

open_alerts = [
    alert
    for alert in alerts
    if alert.get("status") == "open"
]

active_visits = [
    visit
    for visit in visits
    if visit.get("status") == "in_store"
]

flagged_visits = [
    visit
    for visit in visits
    if visit.get("status") == "exited_flagged"
]


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Visits",
        len(visits),
    )


with col2:
    st.metric(
        "Currently Inside",
        len(active_visits),
    )


with col3:
    st.metric(
        "Flagged Visits",
        len(flagged_visits),
    )


with col4:
    st.metric(
        "Open Alerts",
        len(open_alerts),
    )


st.divider()


st.subheader(
    "Recent Visits"
)


if visits:

    dataframe = pd.DataFrame(
        visits
    )

    columns = [
        "track_id",
        "camera_id",
        "entered_at",
        "exited_at",
        "status",
        "unpaid_item_count",
    ]

    available = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    st.dataframe(
        dataframe[available],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No visits recorded yet. "
        "Open Live Monitor and process a video."
    )


st.divider()


st.subheader(
    "🚨 Open Security Alerts"
)


if open_alerts:

    for alert in open_alerts:

        st.warning(
            f"Visit {alert['visit_id'][:8]} — "
            f"{alert['unpaid_item_count']} "
            f"unpaid item(s)"
        )

    st.page_link(
        "pages/3_Theft_Alerts.py",
        label="Open Alert Review",
        icon="🚨",
    )

else:

    st.success(
        "No open security alerts."
    )


st.divider()


st.subheader(
    "How to test"
)

st.markdown(
    """
1. Open **🎥 Live Monitor**.
2. Upload a short store video.
3. Let Store Vision AI detect and track people.
4. Items detected around the shelf zone are recorded.
5. When a tracked person reaches the exit zone,
   the system compares observed items against billing.
6. If something remains unpaid, an **alert is created**.
7. Open **🚨 Security Alerts**.
8. A human staff member confirms or dismisses the alert.
"""
)