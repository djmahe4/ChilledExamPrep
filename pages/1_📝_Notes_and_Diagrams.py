# pages/1_📝_Notes_and_Diagrams.py
import streamlit as st
from utils import (
    load_text_data,
    load_pdf_data,
    extract_and_process_images_from_pdf,
    get_gemini_response
)
import json
import time

st.set_page_config(page_title="Notes & Diagrams", page_icon="📝", layout="wide")
st.title("📝 Notes Management & Diagram Analysis")

# --- Initialize Page-Specific Session State ---
if "nd_current_notes_text" not in st.session_state:
    st.session_state.nd_current_notes_text = None
if "nd_processed_diagrams" not in st.session_state:
    st.session_state.nd_processed_diagrams = []
if "nd_current_filename" not in st.session_state:
    st.session_state.nd_current_filename = None
if "yolo_model_instance" not in st.session_state: # Should be set by Home
    st.session_state.yolo_model_instance = None
if "api_key_configured" not in st.session_state: # Should be set by Home
    st.session_state.api_key_configured = False


# --- Check for API Key and YOLO model ---
if not st.session_state.api_key_configured:
    st.warning("Gemini API Key not configured. Please set it on the 'Study_App_Home' page to use AI features.", icon="🔑")
if not st.session_state.yolo_model_instance:
    st.warning("YOLO Model not loaded. Diagram detection capabilities will be limited. Load it from 'Study_App_Home'.", icon="🖼️")

# --- File Uploader and Processing ---
uploaded_notes_file = st.file_uploader("Upload your notes (TXT or PDF)", type=["txt", "pdf"], key="nd_notes_uploader")

# Sidebar for YOLO parameters for this page
st.sidebar.header("Diagram Detection Tunables")
yolo_conf_thresh_nd = st.sidebar.slider("YOLO Confidence", 0.1, 0.9, 0.25, 0.05, key="yolo_conf_nd")
yolo_min_objs_nd = st.sidebar.slider("Min Objects for Diagram", 1, 10, 3, 1, key="yolo_min_obj_nd")


if uploaded_notes_file is not None:
    if st.button("Process Notes File", key="nd_process_button"):
        st.session_state.nd_current_notes_text = None
        st.session_state.nd_processed_diagrams = []
        st.session_state.nd_current_filename = uploaded_notes_file.name
        
        file_content_bytes = uploaded_notes_file.getvalue()

        if uploaded_notes_file.type == "text/plain":
            st.session_state.nd_current_notes_text = load_text_data(file_content_bytes, uploaded_notes_file.name)
            st.success("Text notes loaded successfully!")
        elif uploaded_notes_file.type == "application/pdf":
            pdf_bytes = load_pdf_data(file_content_bytes, uploaded_notes_file.name) # Returns bytes
            if pdf_bytes:
                # Extract text
                try:
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf") # PyMuPDF needs stream of bytes for type='pdf'
                    full_text = "".join([page.get_text() for page in doc])
                    doc.close()
                    st.session_state.nd_current_notes_text = full_text
                    st.success("PDF text content extracted!")
                except Exception as e:
                    st.error(f"Error extracting text from PDF: {e}")
                    st.session_state.nd_current_notes_text = "Error extracting text."

                # Extract and process images if YOLO model is available
                if st.session_state.yolo_model_instance:
                    with st.spinner("Analyzing images for diagrams (YOLO)..."):
                        # Pass the ID of the model object for caching, the function retrieves model from session_state
                        yolo_model_id = id(st.session_state.yolo_model_instance)
                        st.session_state.nd_processed_diagrams = extract_and_process_images_from_pdf(
                            pdf_bytes,
                            uploaded_notes_file.name, # for _pdf_filename_key
                            yolo_model_id, # for _yolo_model_object_id
                            yolo_confidence_threshold=yolo_conf_thresh_nd,
                            yolo_min_objects_for_diagram=yolo_min_objs_nd
                        )
                    num_diag = sum(1 for d in st.session_state.nd_processed_diagrams if d["is_potential_diagram"])
                    st.success(f"Processed {len(st.session_state.nd_processed_diagrams)} images. Flagged {num_diag} as potential diagrams.")
                else:
                    st.info("YOLO model not loaded, skipping diagram detection for PDF images.")
        st.rerun() # Update UI

# --- Display Notes and Diagrams ---
if st.session_state.nd_current_notes_text:
    st.subheader("Your Notes Content:")
    with st.expander("View Notes Text", expanded=False):
        st.text_area("Notes", st.session_state.nd_current_notes_text, height=200, disabled=True, key="nd_notes_view")
    # (Add Text Skimming UI here if desired, using utils.get_gemini_response)

    if st.session_state.api_key_configured:
        st.subheader("Ask Gemini About Your Notes:")
        notes_query = st.text_input("Enter your question about the notes:", key="nd_notes_q")
        if st.button("Get Answer", key="nd_ask_notes"):
            if notes_query:
                prompt = f"Based on these notes:\n\n{st.session_state.nd_current_notes_text}\n\nAnswer: {notes_query}"
                with st.spinner("Gemini is thinking..."):
                    answer = get_gemini_response(prompt)
                st.markdown("**Answer:**")
                st.info(answer or "Could not get an answer.")
            else:
                st.warning("Please type a question.")

if st.session_state.nd_processed_diagrams:
    st.subheader("🖼️ Extracted Images & Potential Diagrams")
    potential_diagrams_found = False
    for i, data in enumerate(st.session_state.nd_processed_diagrams):
        if data["is_potential_diagram"]:
            potential_diagrams_found = True
            st.markdown(f"--- \n **Potential Diagram {i+1} (Page {data['page']})**")
            st.image(data["image_pil"], caption=f"Objects: {data['yolo_detected_objects']} ({', '.join(data['yolo_detections_summary'])}). OCR: '{data['ocr_text'][:100]}...'")
            if st.session_state.api_key_configured and st.button(f"Explain Diagram {i+1}", key=f"nd_explain_diag_{i}"):
                ocr_context = data['ocr_text'] if data['ocr_text'] and "OCR Error" not in data['ocr_text'] else "No clear OCR text available."
                yolo_context = f"The image was flagged as a diagram with {data['yolo_detected_objects']} objects detected (classes: {', '.join(data['yolo_detections_summary'])})."
                prompt_explain = f"Context: {yolo_context}\nOCR Text from diagram: '{ocr_context}'.\nExplain this diagram or the concepts it likely represents."
                with st.spinner("Gemini is analyzing..."):
                    explanation = get_gemini_response(prompt_explain)
                st.info(explanation or "Could not get an explanation.")
    if not potential_diagrams_found and any(st.session_state.nd_processed_diagrams):
        st.markdown("No images met the criteria to be flagged as 'potential diagrams' based on current settings.")

# --- Download Data for NN ---
if st.session_state.nd_current_notes_text or st.session_state.nd_processed_diagrams:
    st.subheader("⬇️ Download Processed Data")
    nn_data_list = []
    if st.session_state.nd_current_notes_text:
        nn_data_list.append({
            "type": "full_notes_text", "source_file": st.session_state.nd_current_filename or "unknown",
            "content": st.session_state.nd_current_notes_text, "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    for diag_data in st.session_state.nd_processed_diagrams:
        if diag_data["is_potential_diagram"]:
            nn_data_list.append({
                "type": "diagram_info", "source_file": diag_data["filename"], "page": diag_data["page"],
                "ocr_content": diag_data["ocr_text"], "yolo_object_count": diag_data["yolo_detected_objects"],
                "yolo_classes": diag_data["yolo_detections_summary"], "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    if nn_data_list:
        json_data = json.dumps(nn_data_list, indent=4)
        st.download_button(
            label="Download Notes & Diagram Data (JSON)", data=json_data,
            file_name=f"notes_diagram_data_{st.session_state.nd_current_filename.split('.')[0] if st.session_state.nd_current_filename else 'export'}.json",
            mime="application/json", key="nd_download_json"
        )