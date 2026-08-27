import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Supabase Diagnostic")

st.title("🔧 Store Vision AI — Supabase Diagnostic")

# ---------------------------------------------------------
# Read Streamlit Secrets
# ---------------------------------------------------------

url = st.secrets.get("SUPABASE_URL")
key = st.secrets.get("SUPABASE_PUBLISHABLE_KEY")

if not url:
    st.error("SUPABASE_URL is missing from Streamlit Secrets.")
    st.stop()

if not key:
    st.error("SUPABASE_PUBLISHABLE_KEY is missing from Streamlit Secrets.")
    st.stop()

st.success("Supabase Secrets are present.")

# Show only the project URL, NEVER the key
st.write("Supabase URL:")
st.code(url)

# ---------------------------------------------------------
# Connect
# ---------------------------------------------------------

try:
    supabase = create_client(url, key)
    st.success("Supabase client created successfully.")
except Exception as e:
    st.error("Could not create Supabase client.")
    st.exception(e)
    st.stop()

# ---------------------------------------------------------
# READ TEST
# ---------------------------------------------------------

st.subheader("1. Database Read Test")

try:
    result = (
        supabase
        .table("stores")
        .select("id,owner_id,name")
        .limit(5)
        .execute()
    )

    st.success("Successfully read the stores table.")

    if result.data:
        st.dataframe(result.data, use_container_width=True)
    else:
        st.info("stores table is empty.")

except Exception as e:
    st.error("READ FAILED")
    st.exception(e)


# ---------------------------------------------------------
# INSERT TEST
# ---------------------------------------------------------

st.subheader("2. Database Insert Test")

if st.button("Test Store Insert"):

    try:
        result = (
            supabase
            .table("stores")
            .insert({
                "name": "Streamlit Diagnostic Store",
                "address": "Diagnostic Test"
            })
            .execute()
        )

        st.success("INSERT WORKED!")
        st.write(result.data)

    except Exception as e:
        st.error("INSERT FAILED")
        st.exception(e)