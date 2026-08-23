import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .feature-card {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #ddd;
        background: white;
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("👁️ Store Vision AI")
st.sidebar.caption("Retail Computer Vision Platform")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.info(
    "Use the pages in the sidebar to analyse shelf images, "
    "view analytics and generate reports."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Version")
st.sidebar.success("v1.0 MVP")

# ---------- Home ----------
st.markdown('<div class="main-title">👁️ Store Vision AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">'
    "AI-powered retail shelf monitoring and visual inventory analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "Store Vision AI analyses retail shelf images and converts visual information "
    "into useful inventory and shelf-performance insights."
)

st.markdown("## 🚀 What this MVP can do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
        <h3>📷 Image Analysis</h3>
        <p>Upload a shelf image and analyse its visual structure.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
        <h3>📦 Product Analysis</h3>
        <p>Estimate visible products and identify shelf occupancy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
        <h3>📊 Analytics</h3>
        <p>Turn visual analysis into inventory-style metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown("## 🧠 Development Roadmap")

roadmap = [
    ("✅", "Phase 1", "Streamlit dashboard and image analysis"),
    ("🔄", "Phase 2", "YOLO-based product detection"),
    ("⏳", "Phase 3", "Real-time webcam/video analysis"),
    ("⏳", "Phase 4", "Product recognition and SKU detection"),
    ("⏳", "Phase 5", "Stock-out and empty-shelf alerts"),
    ("⏳", "Phase 6", "AI retail assistant"),
]

for icon, phase, description in roadmap:
    st.markdown(f"**{icon} {phase}** — {description}")

st.markdown("---")

st.caption(
    "Store Vision AI | Portfolio Project | Built with Python + Streamlit + Computer Vision"
)