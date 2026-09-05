"""
Store Vision AI - Vision Scanner
"""

import cv2
import pandas as pd
import streamlit as st

from vision_pipeline import (
    cuda_available,
    get_available_profiles,
    process_image,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vision Scanner",
    page_icon="👁️",
    layout="wide",
)


st.title("👁️ Vision Scanner")

st.caption(
    "Analyze store images using the Store Vision AI vision engine."
)


# ============================================================
# PERFORMANCE MODE
# ============================================================

st.subheader("⚡ Inference Mode")


available_profiles = (
    get_available_profiles()
)


profile_labels = [
    profile["label"]
    for profile
    in available_profiles
]


label_to_value = {
    profile["label"]:
        profile["value"]

    for profile
    in available_profiles
}


selected_label = st.selectbox(

    "Choose processing mode",

    profile_labels,

    index=0,

    help=(
        "Auto automatically chooses the best "
        "available hardware."
    ),
)


selected_profile = label_to_value[
    selected_label
]


selected_info = next(

    profile

    for profile
    in available_profiles

    if profile["value"]
    == selected_profile
)


st.info(
    selected_info["help"]
)


if selected_profile == "full_gpu":

    st.success(
        "🚀 CUDA GPU detected — "
        "Full GPU mode is active."
    )


elif selected_profile == "light_cpu":

    if not cuda_available():

        st.caption(
            "CPU mode is active because "
            "no CUDA GPU was detected."
        )


# ============================================================
# CONFIDENCE
# ============================================================

confidence = st.slider(

    "Detection confidence",

    min_value=0.10,

    max_value=0.90,

    value=0.35,

    step=0.05,
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a store image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image_bytes = (
        uploaded_file.getvalue()
    )

    with st.spinner(
        "Running Store Vision AI..."
    ):

        try:

            result = process_image(

                image_bytes=image_bytes,

                confidence=confidence,

                profile=selected_profile,
            )

        except Exception as error:

            st.error(
                f"Vision processing failed: {error}"
            )

            st.stop()


    st.divider()


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Objects Detected",
        result["count"],
    )


    col2.metric(
        "Mode",
        result["profile_label"],
    )


    col3.metric(
        "Model",
        result["model"],
    )


    width, height = result[
        "image_size"
    ]


    col4.metric(
        "Image Size",
        f"{width} × {height}",
    )


    st.caption(

        f"Device: {result['device']} "
        f" • Inference size: "
        f"{result['imgsz']}px"
        f" • Confidence: "
        f"{confidence:.2f}"
    )


    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.subheader(
        "📷 Detection Result"
    )


    display_image = cv2.cvtColor(

        result["image"],

        cv2.COLOR_BGR2RGB,
    )


    st.image(

        display_image,

        caption=(
            "Store Vision AI detection result"
        ),

        use_container_width=True,
    )


    # ========================================================
    # DETECTION TABLE
    # ========================================================

    st.subheader(
        "🔎 Detected Objects"
    )


    if result["detections"]:

        rows = []


        for detection in result[
            "detections"
        ]:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )


            rows.append(

                {
                    "Object":
                        detection[
                            "class_name"
                        ],

                    "Confidence":
                        round(
                            detection[
                                "confidence"
                            ],
                            3,
                        ),

                    "Bounding Box":
                        (
                            f"({x1:.0f}, "
                            f"{y1:.0f}, "
                            f"{x2:.0f}, "
                            f"{y2:.0f})"
                        ),
                }
            )


        dataframe = pd.DataFrame(
            rows
        )


        st.dataframe(

            dataframe,

            use_container_width=True,

            hide_index=True,
        )


        # ====================================================
        # OBJECT SUMMARY
        # ====================================================

        st.subheader(
            "📊 Object Summary"
        )


        summary_rows = [

            {
                "Object":
                    label,

                "Count":
                    count,
            }

            for label, count

            in sorted(

                result[
                    "class_counts"
                ].items(),

                key=lambda item:
                    (-item[1], item[0]),
            )
        ]


        st.dataframe(

            pd.DataFrame(
                summary_rows
            ),

            use_container_width=True,

            hide_index=True,
        )


    else:

        st.warning(

            "No objects were detected. "
            "Try a clearer image or lower "
            "the confidence threshold."
        )


else:

    st.info(

        "Upload a JPG, JPEG, PNG, "
        "or WEBP store image to start scanning."
    )