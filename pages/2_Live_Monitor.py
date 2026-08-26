import os
import tempfile
import time

import streamlit as st

import store_events
import vision_pipeline


st.set_page_config(
    page_title="Live Monitor",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 Store Vision AI — Live Monitor")

st.caption(
    "Upload a short store video. The system detects people, "
    "tracks visits, observes items, compares billing, and flags "
    "unpaid-item exits for human review."
)


with st.sidebar:
    st.header("Store")

    store_name = st.text_input(
        "Store name",
        "Demo Store",
    )

    address = st.text_input(
        "Address",
        "",
    )

    st.header("Detection")

    confidence = st.slider(
        "Confidence",
        0.10,
        0.90,
        0.40,
        0.05,
    )

    frame_stride = st.slider(
        "Process every Nth frame",
        1,
        5,
        2,
    )

    shelf_zone = st.slider(
        "Shelf zone starts",
        0.10,
        0.90,
        0.55,
        0.05,
    )

    exit_zone = st.slider(
        "Exit zone starts",
        0.60,
        0.98,
        0.85,
        0.05,
    )


uploaded = st.file_uploader(
    "Upload store video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
    ],
)


if "processing_log" not in st.session_state:
    st.session_state.processing_log = []


if uploaded:

    st.video(uploaded)

    if st.button(
        "▶ Start Store Vision AI",
        type="primary",
    ):

        try:
            store_id = (
                store_events
                .get_or_create_store(
                    store_name,
                    address,
                )
            )
        except Exception as error:
            st.error(
                "Supabase connection failed."
            )
            st.code(str(error))
            st.stop()

        extension = os.path.splitext(
            uploaded.name
        )[1]

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temporary:

            temporary.write(
                uploaded.read()
            )

            video_path = temporary.name

        frame_area = st.empty()
        status_area = st.empty()
        log_area = st.empty()

        processed = 0
        start_time = time.time()

        st.session_state.processing_log = []

        try:

            for frame, message in (
                vision_pipeline.process_video(
                    video_path,
                    store_id,
                    shelf_zone_frac=shelf_zone,
                    exit_zone_frac=exit_zone,
                    conf=confidence,
                    frame_stride=frame_stride,
                )
            ):

                processed += 1

                frame_area.image(
                    frame,
                    channels="BGR",
                    use_container_width=True,
                )

                if message:
                    st.session_state.processing_log.insert(
                        0,
                        message,
                    )

                elapsed = (
                    time.time()
                    - start_time
                )

                status_area.info(
                    f"Processed frames: {processed} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

                if st.session_state.processing_log:
                    log_area.write(
                        st.session_state.processing_log[
                            :20
                        ]
                    )

            st.success(
                "Video processing completed. "
                "Check the Theft Alerts page."
            )

        except Exception as error:

            st.error(
                "Video processing failed."
            )

            st.exception(error)

        finally:

            if os.path.exists(video_path):
                os.remove(video_path)

else:

    st.info(
        "Upload a short MP4/MOV/AVI video to begin."
    )