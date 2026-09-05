import cv2
import streamlit as st

from vision_pipeline import process_image


st.set_page_config(
    page_title="Vision Scanner",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Vision Scanner")

st.write(
    "Upload a store image and run YOLO object detection."
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
)

uploaded = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)

if uploaded:

    image_bytes = uploaded.read()

    with st.spinner(
        "Running computer vision..."
    ):

        result = process_image(
            image_bytes,
            confidence,
        )

    image = result["image"]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    st.image(
        image_rgb,
        caption="Detection result",
        use_container_width=True,
    )

    st.metric(
        "Objects detected",
        result["count"],
    )

    st.subheader("Detected Objects")

    if result["detections"]:

        rows = []

        for item in result["detections"]:

            rows.append(
                {
                    "Object": item["label"],
                    "Confidence": round(
                        item["confidence"],
                        3,
                    ),
                    "Bounding Box": str(
                        item["bbox"]
                    ),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "No objects detected."
        )