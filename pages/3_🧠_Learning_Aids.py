# pages/3_🧠_Learning_Aids.py
import streamlit as st
from utils import get_gemini_response

st.set_page_config(page_title="Learning Aids", page_icon="🧠", layout="wide")
st.title("🧠 Concept Learning Aids")

# --- Check for API Key ---
if "api_key_configured" not in st.session_state: # Should be set by Home
    st.session_state.api_key_configured = False

if not st.session_state.api_key_configured:
    st.warning("Gemini API Key not configured. Please set it on the 'Study_App_Home' page to use these AI features.", icon="🔑")
    st.stop()

# --- Mnemonic Generation ---
st.subheader("✨ Generate Mnemonics")
concept_for_mnemonic = st.text_area(
    "Enter a concept or piece of information you want to remember:",
    height=100,
    key="la_mnemonic_concept"
)
if st.button("Get Mnemonic Suggestion", key="la_get_mnemonic"):
    if concept_for_mnemonic:
        prompt = f"Generate a few creative and effective mnemonic techniques (like acronyms, acrostics, rhymes, or keyword method) to help remember the following concept: '{concept_for_mnemonic}'. Explain each technique briefly."
        with st.spinner("Generating mnemonics..."):
            mnemonics = get_gemini_response(prompt)
        st.markdown("**Mnemonic Suggestions:**")
        st.info(mnemonics or "Could not generate mnemonics at this time.")
    else:
        st.warning("Please enter a concept.")

st.markdown("---")

# --- Topic Simplification ---
st.subheader("💡 Explain a Complex Topic Simply")
complex_topic_input = st.text_area(
    "Enter a complex topic you want explained simply (e.g., like to a 5-year-old or using a basic analogy):",
    height=100,
    key="la_simplify_topic_input"
)
if st.button("Simplify Topic", key="la_simplify_button"):
    if complex_topic_input:
        prompt = f"Explain the following complex topic in very simple terms, as if explaining to a young child or using a clear, basic analogy: '{complex_topic_input}'"
        with st.spinner("Simplifying the topic..."):
            explanation = get_gemini_response(prompt)
        st.markdown("**Simplified Explanation:**")
        st.success(explanation or "Could not simplify the topic at this time.")
    else:
        st.warning("Please enter a topic to explain.")