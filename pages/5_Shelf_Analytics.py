import streamlit as st

from vision_pipeline import decode_image

from shelf_analyzer import (
    analyze_shelf,
)


st.title("📦 Shelf Analytics")

st.caption(
    "Computer-vision shelf occupancy analysis."
)

uploaded = st.file_uploader(
    "Upload shelf image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)

columns = st.slider(
    "Shelf columns",
    2,
    12,
    6,
)

rows = st.slider(
    "Shelf rows",
    1,
    5,
    2,
)


if uploaded:

    image = decode_image(
        uploaded.read()
    )

    if st.button(
        "Analyze Shelf",
        type="primary",
    ):

        result = analyze_shelf(
            image,
            rows=rows,
            columns=columns,
        )

        st.image(
            result["image"],
            channels="BGR",
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total slots",
            result["total_slots"],
        )

        col2.metric(
            "Occupied",
            result["occupied_slots"],
        )

        col3.metric(
            "Empty",
            result["empty_slots"],
        )

        if result["empty_slots"]:

            st.warning(
                f"{result['empty_slots']} "
                "empty shelf positions detected."
            )

        else:

            st.success(
                "Shelf appears fully occupied."
            )