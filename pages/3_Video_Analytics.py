import tempfile

import streamlit as st

from video_pipeline import process_video


st.title("🎥 Video Analytics")

st.caption(
    "Frame-by-frame computer vision and person tracking."
)

uploaded = st.file_uploader(
    "Upload MP4 video",
    type=["mp4", "mov", "avi"],
)

confidence = st.slider(
    "Detection confidence",
    0.1,
    0.9,
    0.35,
)


if uploaded:

    if st.button(
        "Run Video Analysis",
        type="primary",
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as file:

            file.write(
                uploaded.read()
            )

            video_path = file.name

        frame_area = st.empty()

        stats = st.empty()

        total_frames = 0
        max_people = 0

        for result in process_video(
            video_path,
            confidence,
        ):

            frame_area.image(
                result["frame"],
                channels="BGR",
                use_container_width=True,
            )

            total_frames += 1

            max_people = max(
                max_people,
                result["people_count"],
            )

            stats.metric(
                "Processed frames",
                total_frames,
            )

        st.success(
            "Video analysis completed."
        )

        st.write(
            f"Maximum tracked people: "
            f"{max_people}"
        )