import cv2
import numpy as np
import streamlit as st

from models.shelf_analyzer import (
    ShelfAnalyzer,
)

from shelf_service import (
    save_shelf_scan,
    create_shelf_alert,
)


st.set_page_config(
    page_title="Shelf Analytics",
    page_icon="📦",
    layout="wide",
)


st.title(
    "📦 Shelf Stock Intelligence"
)

st.caption(
    "Detect shelf occupancy, empty zones and stock-out risks."
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
        "Please sign in before using Shelf Analytics."
    )

    st.stop()


# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    rows = st.slider(
        "Shelf rows",
        1,
        8,
        4,
    )

with col2:

    columns = st.slider(
        "Shelf columns",
        2,
        12,
        6,
    )


uploaded = st.file_uploader(
    "Upload shelf image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
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
            "Could not read the uploaded image."
        )

        st.stop()


    if st.button(
        "🔎 Analyze Shelf",
        type="primary",
    ):

        analyzer = ShelfAnalyzer(
            rows=rows,
            columns=columns,
        )

        zones = analyzer.analyse(
            image
        )

        summary = analyzer.summary(
            zones
        )

        annotated = analyzer.draw(
            image,
            zones,
        )


        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            ),
            caption="Shelf occupancy analysis",
            use_container_width=True,
        )


        st.divider()


        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Shelf zones",
            summary["total_zones"],
        )

        c2.metric(
            "Occupied",
            (
                summary["total_zones"]
                -
                summary["empty_zones"]
            ),
        )

        c3.metric(
            "Empty",
            summary["empty_zones"],
        )

        c4.metric(
            "Average occupancy",
            f"{summary['average_occupancy']:.1%}",
        )


        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        empty = summary[
            "empty_zones"
        ]

        low = summary[
            "low_zones"
        ]


        if empty > 0:

            st.error(
                f"⚠️ {empty} empty shelf "
                "zone(s) detected."
            )

        elif low > 0:

            st.warning(
                f"⚠️ {low} low-stock "
                "zone(s) detected."
            )

        else:

            st.success(
                "✓ Shelf occupancy looks healthy."
            )


        # -------------------------------------------------
        # Database
        # -------------------------------------------------

        try:

            save_shelf_scan(
                supabase,
                user_id,
                uploaded.name,
                summary,
            )

            st.success(
                "Shelf scan saved to Supabase."
            )

        except Exception as exc:

            st.warning(
                f"Shelf scan could not be saved: {exc}"
            )


        # -------------------------------------------------
        # Alert
        # -------------------------------------------------

        if empty > 0:

            try:

                create_shelf_alert(
                    supabase,
                    user_id,
                    empty,
                )

                st.warning(
                    "🚨 Stock-out alert created."
                )

            except Exception as exc:

                st.warning(
                    f"Alert could not be created: {exc}"
                )


        # -------------------------------------------------
        # Zone table
        # -------------------------------------------------

        st.subheader(
            "Shelf Zone Details"
        )

        for zone in zones:

            st.write(
                f"Row {zone.row}, "
                f"Column {zone.column} — "
                f"**{zone.status}** — "
                f"{zone.occupancy:.1%}"
            )