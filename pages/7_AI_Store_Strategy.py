import streamlit as st

from backend.database import get_table
from retail_assistant import RetailAssistant


st.set_page_config(
    page_title="AI Store Strategy",
    page_icon="🧠",
    layout="wide",
)


st.title("🧠 AI Retail Store Assistant")

st.caption(
    "Computer Vision + Visitor Behaviour + Product Analytics"
)


products = get_table(
    "products"
)

sales = get_table(
    "sales"
)

visits = get_table(
    "store_visits"
)

events = get_table(
    "retail_attraction_events"
)


assistant = RetailAssistant(
    products=products,
    sales=sales,
    visits=visits,
    attraction_events=events,
)


summary = assistant.summary()


# ============================================================
# SUMMARY
# ============================================================

c1, c2, c3 = st.columns(3)


c1.metric(
    "Visitors",
    summary["visitor_count"]
)


c2.metric(
    "Products analysed",
    len(
        summary[
            "attraction_scores"
        ]
    )
)


c3.metric(
    "Recommendations",
    len(
        summary[
            "recommendations"
        ]
    )
)


# ============================================================
# PRODUCT ATTRACTION
# ============================================================

st.subheader(
    "🔥 Most Attractive Products"
)


scores = summary[
    "attraction_scores"
]


if scores:

    ranking = sorted(
        scores.items(),
        key=lambda x:
            x[1]["attraction_score"],
        reverse=True,
    )

    for label, stats in ranking:

        st.write(
            f"### {label}"
        )

        st.write(
            f"Attraction score: "
            f"**{stats['attraction_score']:.0f}**"
        )

        st.write(
            f"Approaches: {stats['approaches']}  |  "
            f"Inspections: {stats['inspects']}  |  "
            f"Picks: {stats['picks']}  |  "
            f"Returns: {stats['returns']}"
        )

else:

    st.info(
        "No attraction events have been collected yet."
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.subheader(
    "💡 Store Recommendations"
)


recommendations = (
    summary["recommendations"]
)


if not recommendations:

    st.info(
        "The assistant needs more visitor/product "
        "interaction data before generating strong "
        "recommendations."
    )

else:

    for recommendation in recommendations:

        priority = recommendation[
            "priority"
        ]

        if priority == "high":

            st.error(
                recommendation["title"]
            )

        else:

            st.warning(
                recommendation["title"]
            )

        st.write(
            recommendation[
                "recommendation"
            ]
        )

        with st.expander(
            "View evidence"
        ):

            st.json(
                recommendation[
                    "evidence"
                ]
            )