# pages/2_📜_Question_Papers.py
import streamlit as st
from utils import load_text_data, load_pdf_data, get_gemini_response
import pymupdf # For PDF text extraction directly here as it's simple

st.set_page_config(page_title="Question Papers", page_icon="📜", layout="wide")
st.title("📜 Question Papers Chat")

# --- Initialize Page-Specific Session State ---
if "qp_current_text" not in st.session_state:
    st.session_state.qp_current_text = None
if "qp_current_filename" not in st.session_state:
    st.session_state.qp_current_filename = None
if "qp_chat_messages" not in st.session_state:
    st.session_state.qp_chat_messages = []
if "api_key_configured" not in st.session_state: # Should be set by Home
    st.session_state.api_key_configured = False

# --- Check for API Key ---
if not st.session_state.api_key_configured:
    st.warning("Gemini API Key not configured. Please set it on the 'Study_App_Home' page to use AI features.", icon="🔑")
    st.stop() # Block further execution on this page if API key isn't set

# --- File Uploader and Processing ---
uploaded_qp_file = st.file_uploader("Upload Question Paper (TXT or PDF)", type=["txt", "pdf"], key="qp_uploader_page")

if uploaded_qp_file is not None:
    if st.button("Process Question Paper File", key="qp_process_button"):
        st.session_state.qp_current_text = None # Reset
        st.session_state.qp_chat_messages = []  # Reset chat on new file
        st.session_state.qp_current_filename = uploaded_qp_file.name
        
        file_content_bytes = uploaded_qp_file.getvalue()

        if uploaded_qp_file.type == "text/plain":
            st.session_state.qp_current_text = load_text_data(file_content_bytes, uploaded_qp_file.name)
        elif uploaded_qp_file.type == "application/pdf":
            pdf_bytes = load_pdf_data(file_content_bytes, uploaded_qp_file.name)
            if pdf_bytes:
                try:
                    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    st.session_state.qp_current_text = "".join([page.get_text() for page in doc])
                    doc.close()
                except Exception as e:
                    st.error(f"Error extracting text from PDF: {e}")
                    st.session_state.qp_current_text = "Error extracting text."
        
        if st.session_state.qp_current_text and "Error" not in st.session_state.qp_current_text:
            st.success("Question paper processed and ready for chat!")
        else:
            st.error("Could not process the question paper.")
        st.rerun()

# --- Display QP and Chat ---
if st.session_state.qp_current_text:
    st.subheader(f"Chatting about: {st.session_state.qp_current_filename}")
    with st.expander("View Question Paper Text", expanded=False):
        st.text_area("Question Paper Content", st.session_state.qp_current_text, height=200, disabled=True, key="qp_text_view")

    # Chat interface
    for message in st.session_state.qp_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the paper...")
    if user_query:
        st.session_state.qp_chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Gemini is thinking..."):
            prompt = f"Based on this question paper:\n\n{st.session_state.qp_current_text}\n\nUser's question: {user_query}\n\nAnswer:"
            response = get_gemini_response(prompt)
            with st.chat_message("assistant"):
                st.markdown(response or "Sorry, I couldn't get a response.")
            st.session_state.qp_chat_messages.append({"role": "assistant", "content": response or "Error."})
else:
    st.info("Upload a question paper to begin.")
