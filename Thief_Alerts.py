"""
Theft Alerts page — human-review queue. The vision pipeline flags exits
with unpaid items; a person always makes the final call here. This
review step exists on purpose: automated flags have false positives
(returned items, self-checkout apps, etc.), and a system that skips
straight to an accusation is both bad UX and a liability.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import store_events

st.set_page_config(page_title="Theft Alerts · Store Vision AI", page_icon="🚨", layout="wide")

st.title("🚨 Flagged Visits")
st.caption("Open items require a staff decision: confirm or dismiss as a false positive.")

reviewer = st.sidebar.text_input("Reviewing as", value="staff")

try:
    open_alerts = store_events.get_open_alerts()
except Exception as e:
    st.error(f"Couldn't reach Supabase: {e}")
    st.stop()

if not open_alerts:
    st.success("No open alerts. Nothing waiting on review.")
else:
    for alert in open_alerts:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**Visit** `{alert['visit_id'][:8]}` — track flagged at exit")
                st.caption(
                    f"{alert['unpaid_item_count']} item(s) detected near the shelf with no "
                    f"matching billing event · opened {alert['created_at']}"
                )
            with c2:
                if st.button("✅ Confirm", key=f"confirm_{alert['id']}"):
                    store_events.review_alert(alert["id"], "confirmed", reviewer)
                    st.rerun()
            with c3:
                if st.button("❌ Dismiss", key=f"dismiss_{alert['id']}"):
                    store_events.review_alert(alert["id"], "dismissed", reviewer)
                    st.rerun()

st.markdown("---")
st.markdown("### Recent history")

try:
    history = store_events.get_all_alerts(limit=100)
except Exception:
    history = []

if history:
    df = pd.DataFrame(history)[["created_at", "status", "unpaid_item_count", "reviewed_by", "notes"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    status_counts = df["status"].value_counts()
    st.bar_chart(status_counts)
else:
    st.caption("No alert history yet.")
