"""
Live Monitor page — upload a video (or point to a sample clip), run the
tracking + zone pipeline over it, and watch annotated frames and events
stream in. Streamlit Cloud's free tier is CPU-only and RAM-limited, so
this processes uploaded/sample footage rather than a true live camera
feed — a short clip at a lowered frame rate keeps it responsive.
"""
import os
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import store_events
import vision_pipeline

st.set_page_config(page_title="Live Monitor · Store Vision AI", page_icon="🎥", layout="wide")

st.title("🎥 Live Monitor")
st.caption(
    "Tracks people across a shelf zone and an exit zone. If a tracked person "
    "leaves with more items detected than billed, the visit is flagged for "
    "staff review — not labeled as theft outright."
)

with st.sidebar:
    st.markdown("### Store")
    store_name = st.text_input("Store name", value="Demo Store")

    st.markdown("### Zones")
    st.caption("Vertical bands across the frame width, left to right.")
    shelf_frac = st.slider("Shelf zone starts at", 0.1, 0.9, 0.55, 0.05)
    exit_frac = st.slider("Exit zone starts at", shelf_frac + 0.05, 0.98, max(0.85, shelf_frac + 0.1), 0.05)

    st.markdown("### Performance")
    conf = st.slider("Detection confidence", 0.1, 0.9, 0.4, 0.05)
    frame_stride = st.slider("Process every Nth frame", 1, 5, 2)

uploaded = st.file_uploader("Upload a short video clip (mp4/mov)", type=["mp4", "mov", "avi", "mkv"])

col_video, col_log = st.columns([2, 1])
frame_slot = col_video.empty()
metric_slot = col_video.empty()
log_slot = col_log.container(height=480)

if "sv_log" not in st.session_state:
    st.session_state.sv_log = []

start = st.button("▶ Start processing", type="primary", disabled=uploaded is None)

if start and uploaded is not None:
    try:
        store_id = store_events.get_or_create_store(store_name)
    except Exception as e:
        st.error(
            "Couldn't reach Supabase. Check SUPABASE_URL / SUPABASE_KEY in your "
            f"secrets. ({e})"
        )
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.session_state.sv_log = []
    flagged_count = 0
    processed = 0
    t0 = time.time()

    try:
        for frame, log_line in vision_pipeline.process_video(
            video_path,
            store_id,
            shelf_zone_frac=shelf_frac,
            exit_zone_frac=exit_frac,
            conf=conf,
            frame_stride=frame_stride,
        ):
            processed += 1
            frame_slot.image(frame, channels="BGR", use_container_width=True)

            if log_line:
                st.session_state.sv_log.insert(0, log_line)
                if "flagged for review" in log_line:
                    flagged_count += 1

            metric_slot.markdown(
                f"**Frames processed:** {processed}  ·  "
                f"**Flagged exits:** {flagged_count}  ·  "
                f"**Elapsed:** {time.time() - t0:.1f}s"
            )
            with log_slot:
                for line in st.session_state.sv_log[:30]:
                    st.text(line)
    finally:
        os.unlink(video_path)

    st.success("Done processing this clip. Check the Theft Alerts page for anything flagged.")

elif not uploaded:
    st.info("Upload a clip to get started. A short video of someone walking past a desk with a bag or bottle works fine for testing zones.")
