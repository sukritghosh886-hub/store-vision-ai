import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Allow imports from project root
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.detector import ShelfDetector
from models.shelf_analyzer import ShelfAnalyzer
from utils.image_utils import (
    image_statistics,
    opencv_to_pil,
    pil_to_opencv,
    resize_image,
    uploaded_file_to_pil,
)


st.set_page_config(
    page_title="Image Analysis | Store Vision AI",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Shelf Image Analysis")
st.caption(
    "Upload a retail shelf image and analyse its visual structure."
)

# ---------- Sidebar ----------
st.sidebar.header("Analysis Settings")

rows = st.sidebar.slider(
    "Shelf rows",
    min_value=1,
    max_value=8,
    value=4,
)

columns = st.sidebar.slider(
    "Shelf columns",
    min_value=1,
    max_value=12,
    value=6,
)

minimum_area = st.sidebar.slider(
    "Minimum object area",
    min_value=100,
    max_value=5000,
    value=800,
    step=100,
)

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload a shelf image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info(
        "👆 Upload a JPG, JPEG, PNG or WEBP shelf image to begin."
    )

    st.markdown(
        """
        ### Example workflow

        **1. Upload image → 2. Detect visual objects → "
        "3. Analyse shelf zones → 4. View metrics → 5. Export results**

        The current version uses OpenCV as a baseline.
        A trained YOLO product detector will be added in the next phase.
        """
    )

    st.stop()

# ---------- Load image ----------
try:
    original_image = uploaded_file_to_pil(
        uploaded_file
    )

    display_image = resize_image(
        original_image
    )

except Exception as error:
    st.error(
        f"Unable to read the image: {error}"
    )
    st.stop()

# ---------- Basic statistics ----------
stats = image_statistics(
    original_image
)

# ---------- Models ----------
detector = ShelfDetector(
    min_area=minimum_area
)

analyzer = ShelfAnalyzer(
    rows=rows,
    columns=columns,
)

opencv_image = pil_to_opencv(
    display_image
)

# ---------- Detection ----------
with st.spinner("Analysing image..."):

    detections = detector.detect(
        opencv_image
    )

    detection_image = detector.draw_detections(
        opencv_image,
        detections,
    )

    zones = analyzer.analyse(
        opencv_image
    )

    summary = analyzer.summary(
        zones
    )

# ---------- Metrics ----------
st.markdown("## 📊 Analysis Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Detected Objects",
        len(detections),
    )

with col2:
    st.metric(
        "Average Shelf Occupancy",
        f"{summary['average_occupancy']:.1%}",
    )

with col3:
    st.metric(
        "Empty Zones",
        summary["empty_zones"],
    )

with col4:
    st.metric(
        "Image Size",
        f"{stats['width']} × {stats['height']}",
    )

# ---------- Images ----------
st.markdown("## 🔍 Visual Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Original Image")

    st.image(
        display_image,
        use_container_width=True,
    )

with right:
    st.subheader("Detected Objects")

    result_image = opencv_to_pil(
        detection_image
    )

    st.image(
        result_image,
        use_container_width=True,
    )

# ---------- Shelf heatmap ----------
st.markdown("## 🗺️ Shelf Occupancy Map")

heatmap = analyzer.create_heatmap(
    zones
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

image = ax.imshow(
    heatmap,
    vmin=0,
    vmax=1,
)

ax.set_title(
    "Estimated Shelf Occupancy"
)

ax.set_xlabel(
    "Shelf Column"
)

ax.set_ylabel(
    "Shelf Row"
)

ax.set_xticks(
    range(columns)
)

ax.set_yticks(
    range(rows)
)

plt.colorbar(
    image,
    ax=ax,
    label="Occupancy"
)

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)

# ---------- Zone table ----------
st.markdown("## 📋 Shelf Zone Details")

zone_data = []

for zone in zones:
    zone_data.append(
        {
            "Row": zone.row,
            "Column": zone.column,
            "Occupancy": f"{zone.occupancy:.1%}",
            "Status": zone.status,
        }
    )

zone_df = pd.DataFrame(
    zone_data
)

st.dataframe(
    zone_df,
    use_container_width=True,
    hide_index=True,
)

# ---------- Detection table ----------
if detections:

    st.markdown("## 📦 Detected Objects")

    detection_data = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        x, y, width, height = detection.bbox

        detection_data.append(
            {
                "Object": index,
                "Type": detection.label,
                "Confidence": f"{detection.confidence:.1%}",
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height,
            }
        )

    detection_df = pd.DataFrame(
        detection_data
    )

    st.dataframe(
        detection_df,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        "No objects were detected with the current "
        "minimum-area setting. Try reducing the "
        "minimum object area in the sidebar."
    )

# ---------- Image information ----------
with st.expander("ℹ️ Image Information"):

    st.write(
        {
            "Width": stats["width"],
            "Height": stats["height"],
            "Brightness": round(
                stats["brightness"],
                2,
            ),
            "Contrast": round(
                stats["contrast"],
                2,
            ),
        }
    )

st.markdown("---")

st.caption(
    "Store Vision AI v1.0 — Baseline computer vision analysis"
)