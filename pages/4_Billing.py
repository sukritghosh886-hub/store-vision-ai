import streamlit as st

import store_events


st.set_page_config(
    page_title="Billing",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Billing / POS")

st.caption(
    "Use this page to simulate billing events during MVP testing."
)


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
        "Supabase connection failed."
    )

    st.exception(error)

    st.stop()


visit_id = st.text_input(
    "Visit ID",
)


item_label = st.selectbox(
    "Billed item",
    [
        "bottle",
        "backpack",
        "handbag",
        "cup",
        "cell_phone",
        "banana",
        "apple",
        "orange",
    ],
)


quantity = st.number_input(
    "Quantity",
    min_value=1,
    max_value=100,
    value=1,
)


if st.button(
    "💳 Record Billing",
    type="primary",
):

    if not visit_id.strip():

        st.error(
            "Enter a visit ID."
        )

    else:

        try:

            store_events.log_billing_event(
                visit_id=visit_id.strip(),
                item_label=item_label,
                quantity=int(quantity),
                source="manual",
            )

            st.success(
                f"Recorded {quantity} × "
                f"{item_label} as billed."
            )

        except Exception as error:

            st.error(
                "Could not record billing event."
            )

            st.exception(error)