import streamlit as st
import cv2
import numpy as np

from sku_recognition import (
    recognize_sku,
)

from services.sku_service import (
    load_sku_references,
)


st.set_page_config(
    page_title="SKU Recognition",
    page_icon="🏷️",
    layout="wide",
)


st.title("🏷️ Product Recognition & SKU Detection")

st.caption(
    "Match a product image against registered SKU reference images."
)


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

user_id = st.session_state.get(
    "user_id"
)

supabase = st.session_state.get(
    "supabase"
)


if not user_id or supabase is None:

    st.warning(
        "Please sign in before using SKU recognition."
    )

    st.stop()


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "Upload product image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


threshold = st.slider(
    "Minimum match confidence",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
)


if uploaded:

    image_bytes = uploaded.read()

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        st.error(
            "Unable to decode the image."
        )

        st.stop()

    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        ),
        caption="Query image",
        use_container_width=True,
    )


    if st.button(
        "🔍 Recognize Product",
        type="primary",
    ):

        with st.spinner(
            "Loading SKU references..."
        ):

            references = (
                load_sku_references(
                    supabase,
                    user_id,
                )
            )

        if not references:

            st.warning(
                "No SKU reference images are registered yet."
            )

            st.info(
                "Add products and reference images before recognition."
            )

            st.stop()


        with st.spinner(
            "Matching product..."
        ):

            result = recognize_sku(
                image,
                references,
                threshold,
            )


        st.divider()

        if result["matched"]:

            st.success(
                "✓ Product recognized"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Product",
                result["name"],
            )

            col2.metric(
                "SKU",
                result["sku"],
            )

            col3.metric(
                "Confidence",
                f"{result['confidence']:.1%}",
            )

            if result.get("category"):

                st.write(
                    f"**Category:** "
                    f"{result['category']}"
                )

        else:

            st.warning(
                result["message"]
            )

            st.metric(
                "Best similarity",
                f"{result['confidence']:.1%}",
            )