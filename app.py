import streamlit as st

# ============================================================
# Store Vision AI
# ============================================================

st.set_page_config(
    page_title="Store Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #111827 !important;
    }

    /* Subtitle */
    .subtitle {
        font-size: 18px;
        color: #4b5563 !important;
        margin-bottom: 30px;
    }

    /* Feature cards */
    .feature-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #d1d5db;
        background-color: #ffffff !important;
        margin-bottom: 15px;
        min-height: 150px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    /* Feature card heading */
    .feature-card h3 {
        color: #111827 !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        margin-top: 0;
        margin-bottom: 10px;
    }

    /* Feature card description */
    .feature-card p {
        color: #4b5563 !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        opacity: 1 !important;
        margin-bottom: 0;
    }

    /* Keep all card content visible */
    .feature-card * {
        opacity: 1 !important;
    }

    /* Metric cards */
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background-color: #ffffff !important;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👁️ Store Vision AI")
st.sidebar.caption("Retail Computer Vision Platform")

st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation")

st.sidebar.info(
    "Use the dashboard to analyse shelf images, "
    "view product information and explore analytics."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Version")

st.sidebar.success("v1.0 MVP")

# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👁️ Store Vision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered retail shelf monitoring and visual inventory analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.info(
    "Store Vision AI analyses retail shelf images and converts "
    "visual information into useful inventory and shelf-performance insights."
)

# ============================================================
# FEATURES
# ============================================================

st.markdown("## 🚀 What this MVP can do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <h3>📷 Image Analysis</h3>
            <p>
                Upload a shelf image and analyse its visual structure.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <h3>📦 Product Analysis</h3>
            <p>
                Estimate visible products and identify shelf occupancy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <h3>📊 Analytics</h3>
            <p>
                Turn visual analysis into useful inventory-style metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# ROADMAP
# ============================================================

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
    st.markdown(
        f"**{icon} {phase}** — {description}"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Store Vision AI | Portfolio Project | "
    "Built with Python + Streamlit + Computer Vision"
)