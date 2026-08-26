import pandas as pd
import streamlit as st

import store_events


st.set_page_config(
    page_title="Security Alerts",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Security Alerts")

st.caption(
    "These are security flags for staff review. "
    "The system does not automatically declare that a person committed theft."
)


reviewer = st.sidebar.text_input(
    "Reviewer name",
    "Store Manager",
)


try:
    alerts = store_events.get_open_alerts()

except Exception as error:

    st.error(
        "Could not connect to Supabase."
    )

    st.exception(error)

    st.stop()


if not alerts:

    st.success(
        "No open security alerts."
    )

else:

    st.warning(
        f"{len(alerts)} alert(s) require review."
    )

    for alert in alerts:

        with st.container(
            border=True
        ):

            st.subheader(
                "⚠️ Possible unpaid item"
            )

            st.write(
                f"Visit ID: `{alert['visit_id']}`"
            )

            st.write(
                f"Unpaid items detected: "
                f"**{alert['unpaid_item_count']}**"
            )

            st.write(
                f"Created: {alert['created_at']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "✅ Confirm",
                    key=f"confirm_{alert['id']}",
                ):

                    store_events.review_alert(
                        alert["id"],
                        "confirmed",
                        reviewer,
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "❌ Dismiss",
                    key=f"dismiss_{alert['id']}",
                ):

                    store_events.review_alert(
                        alert["id"],
                        "dismissed",
                        reviewer,
                    )

                    st.rerun()


st.divider()

st.subheader(
    "Alert History"
)


try:

    history = (
        store_events
        .get_all_alerts(
            limit=100
        )
    )

except Exception:

    history = []


if history:

    dataframe = pd.DataFrame(
        history
    )

    columns = [
        "created_at",
        "status",
        "unpaid_item_count",
        "reviewed_by",
        "notes",
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
        "No alert history available."
    )