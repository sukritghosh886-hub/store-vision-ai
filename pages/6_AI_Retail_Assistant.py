import streamlit as st

from backend.retail_assistant import (
    RetailAssistant,
)

from backend.supabase_client import (
    create_supabase_client,
)


st.set_page_config(
    page_title="AI Retail Assistant",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title(
    "🤖 AI Retail Assistant"
)

st.caption(
    "Ask natural-language questions about inventory, "
    "sales, shelves, visitors and security alerts."
)


# ---------------------------------------------------------
# Supabase
# ---------------------------------------------------------

try:

    supabase = create_supabase_client()

except Exception as exc:

    st.error(
        "Supabase connection failed."
    )

    st.exception(exc)

    st.stop()


assistant = RetailAssistant(
    supabase
)


# ---------------------------------------------------------
# Example questions
# ---------------------------------------------------------

st.markdown(
    "### Try asking"
)

examples = [
    "Which products are low in stock?",
    "What should I reorder?",
    "What are my sales and revenue?",
    "Which products are best selling?",
    "Do I have any open alerts?",
    "Are there empty shelves?",
    "How many visitors have we tracked?",
    "Give me a business overview.",
]


cols = st.columns(2)

for index, example in enumerate(
    examples
):

    with cols[index % 2]:

        if st.button(
            example,
            use_container_width=True,
            key=f"example_{index}",
        ):

            st.session_state[
                "retail_question"
            ] = example


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Ask your retail assistant",
    value=st.session_state.get(
        "retail_question",
        "",
    ),
    placeholder=(
        "Example: Which products should I reorder?"
    ),
)


if st.button(
    "Ask Assistant",
    type="primary",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analyzing your retail data..."
        ):

            try:

                answer = assistant.answer(
                    question
                )

                st.session_state[
                    "last_answer"
                ] = answer

            except Exception as exc:

                st.error(
                    "The assistant could not "
                    "complete the query."
                )

                st.exception(exc)


# ---------------------------------------------------------
# Answer
# ---------------------------------------------------------

if st.session_state.get(
    "last_answer"
):

    st.divider()

    st.markdown(
        "### Assistant"
    )

    st.markdown(
        st.session_state[
            "last_answer"
        ]
    )

    st.caption(
        "Answers are generated from the "
        "current Supabase retail data."
    )