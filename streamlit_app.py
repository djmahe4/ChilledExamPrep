import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import json
import time
import functools
from ultralytics import YOLO # For YOLO object detection
import numpy as np # For image conversion for YOLO

# --- Configuration and Caching ---

# Cache the API key
@st.cache_resource
def configure_api(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return False

# Cache the YOLO model
@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"): # yolov8n.pt is a small, general model
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model '{model_name}': {e}. Ensure it's a valid model name and 'ultralytics' is installed.")
        return None

# Cache loaded data (notes, questions)
@st.cache_data
def load_text_data(uploaded_file_content, filename): # Pass content and a unique identifier
    if uploaded_file_content is not None:
        return uploaded_file_content.decode()
    return None

@st.cache_data
def load_pdf_data(uploaded_file_content, filename): # Pass content and a unique identifier
    if uploaded_file_content is not None:
        return uploaded_file_content
    return None

# Cache processed PDF images and associated data (including YOLO results and OCR)
@st.cache_data
def extract_and_process_images_from_pdf(pdf_bytes, _pdf_filename_key, yolo_model, ocr_enabled=True, yolo_confidence_threshold=0.25, yolo_min_objects_for_diagram=3):
    processed_images_data = []
    if not yolo_model:
        st.warning("YOLO model not loaded, diagram detection will be skipped.")
        return processed_images_data

    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image_data = pdf_document.extract_image(xref)
                image_bytes = base_image_data["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # YOLO needs RGB

                # Run YOLO detection
                yolo_results = yolo_model(pil_image, conf=yolo_confidence_threshold, verbose=False) # verbose=False to reduce console spam
                detected_objects_count = len(yolo_results[0].boxes)
                is_potential_diagram = detected_objects_count >= yolo_min_objects_for_diagram

                ocr_text = "OCR not performed (not deemed a diagram or OCR disabled)."
                if is_potential_diagram and ocr_enabled:
                    try:
                        ocr_text = pytesseract.image_to_string(pil_image)
                    except Exception as ocr_err:
                        ocr_text = f"OCR Error: {ocr_err}"
                elif not is_potential_diagram:
                    ocr_text = "OCR not performed (image did not meet diagram criteria by YOLO)."


                processed_images_data.append({
                    "page": page_num + 1,
                    "image_pil": pil_image, # Store PIL image for display
                    "ocr_text": ocr_text,
                    "filename": _pdf_filename_key,
                    "yolo_detected_objects": detected_objects_count,
                    "is_potential_diagram": is_potential_diagram,
                    "yolo_detections_summary": [ # Store class names detected
                        yolo_model.names[int(cls)] for cls in yolo_results[0].boxes.cls
                    ] if yolo_results and yolo_results[0].boxes else []
                })
        pdf_document.close()
    except Exception as e:
        st.error(f"Error processing PDF for images/YOLO/OCR: {e}")
    return processed_images_data

# --- Gemini API Interaction with Retry ---
def retry(retries=3, delay=5, backoff=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            while _retries > 0:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _retries -= 1
                    if _retries == 0:
                        st.error(f"Gemini API Error after multiple retries: {e}")
                        # Check for specific API key errors if possible by inspecting 'e'
                        if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) or "API key not valid" in str(e):
                            st.error("Critical API Error: Please ensure your Gemini API key is correct and has the necessary permissions.")
                        return None # Or raise
                    st.warning(f"Gemini API attempt failed, {_retries} retries remaining. Retrying in {_delay}s... Error: {e}")
                    time.sleep(_delay)
                    _delay *= backoff
            return None # Should be unreachable if retries run out and error is not re-raised
        return wrapper
    return decorator

@retry(retries=3, delay=5)
def get_gemini_response(prompt_text, model_name="gemini-1.5-flash-latest"):
    #Rudimentary check for prompt length, though 1.5 Flash has a large limit.
    #A more sophisticated approach would involve tiktoken or similar for precise token counting.
    if len(prompt_text) > 700000: # Approx 700k chars as a loose proxy for too many tokens for many models (Flash has 1M token limit, but safety)
        st.warning("Prompt is very long. Consider summarizing or focusing your query.")
        # Potentially truncate or ask user to shorten, or implement chunking strategy.
        # For now, we'll still send it, relying on Flash's large context.
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt_text)
    return response.text


# --- UI Components (Pomodoro, Mindmap, Skimming - largely unchanged, ensure session state keys are distinct) ---

def display_pomodoro():
    st.subheader("🍅 Pomodoro Timer")
    # Initialize session state variables safely
    if 'pomo_timer_active' not in st.session_state:
        st.session_state.pomo_timer_active = False
    if 'pomo_timer_seconds' not in st.session_state:
        st.session_state.pomo_timer_seconds = 25 * 60
    if 'pomo_break_time' not in st.session_state:
        st.session_state.pomo_break_time = False
    if 'model_name' not in st.session_state:
            st.session_state.model_name="diagram_det.pt"
    # Button to start/pause work session
    if st.button("Start/Pause Pomodoro (25 min Work)" if not st.session_state.pomo_timer_active else "Pause Pomodoro", key="pomodoro_toggle_work"):
        st.session_state.pomo_timer_active = not st.session_state.pomo_timer_active
        if not st.session_state.pomo_timer_active: # If paused
            st.session_state.pomo_break_time = False # Reset break if work paused
        else: # If started/resumed
             if st.session_state.pomo_timer_seconds == 0 : # if starting a fresh session after break or initial
                 st.session_state.pomo_timer_seconds = 25 * 60
                 st.session_state.pomo_break_time = False


    if st.session_state.pomo_timer_active and not st.session_state.pomo_break_time:
        work_placeholder = st.empty()
        # This loop will run on each interaction, not continuously as a background thread
        # For a live timer in Streamlit, more complex solutions like threading with queue or frontend JS are needed.
        # This is a simplified version for demonstration.
        if st.session_state.pomo_timer_seconds > 0:
            mins, secs = divmod(st.session_state.pomo_timer_seconds, 60)
            time_str = '{:02d}:{:02d}'.format(mins, secs)
            work_placeholder.metric("Work Time Remaining", time_str)
            # To make it "tick", we'd need to st.rerun() and decrement, but that can be disruptive.
            # A button to "Tick Timer" or automatic rerun every second (costly) would be an option.
            # For now, it updates on interaction.

            # Simulate a tick (if a button triggers rerun or page interaction happens)
            # For actual automatic timer, a thread with st.experimental_rerun or a JS component is better.
            # This example keeps it simpler: time decreases only when other Streamlit actions cause reruns
            # or if you add a specific mechanism.

        if st.session_state.pomo_timer_seconds == 0 and not st.session_state.pomo_break_time : # Work finished
            st.session_state.pomo_timer_active = False
            st.session_state.pomo_break_time = True
            st.session_state.pomo_timer_seconds = 5 * 60  # Reset for break
            st.success("Work session complete! Take a 5-minute break.")
            st.balloons()
            st.rerun() # Force rerun to switch to break mode UI

    # Break timer logic
    if st.session_state.pomo_break_time:
        if st.button("Start/Pause Break (5 min)" if not st.session_state.pomo_timer_active else "Pause Break", key="pomodoro_toggle_break"):
            st.session_state.pomo_timer_active = not st.session_state.pomo_timer_active
            if st.session_state.pomo_timer_active and st.session_state.pomo_timer_seconds == 0 : # starting break after work or paused break
                st.session_state.pomo_timer_seconds = 5 * 60


        if st.session_state.pomo_timer_active and st.session_state.pomo_break_time:
            break_placeholder = st.empty()
            if st.session_state.pomo_timer_seconds > 0:
                mins, secs = divmod(st.session_state.pomo_timer_seconds, 60)
                time_str = '{:02d}:{:02d}'.format(mins, secs)
                break_placeholder.metric("Break Time Remaining", time_str)

            if st.session_state.pomo_timer_seconds == 0 : # Break finished
                st.session_state.pomo_timer_active = False
                st.session_state.pomo_break_time = False
                st.session_state.pomo_timer_seconds = 25 * 60  # Reset for work
                st.info("Break over! Time for the next Pomodoro.")
                st.rerun()

    # Manual tick button (optional, demonstrates update mechanism)
    if st.session_state.pomo_timer_active:
        if st.button("Tick Timer Manually", key="tick_pomo"):
            if st.session_state.pomo_timer_seconds > 0:
                st.session_state.pomo_timer_seconds -=1
            st.rerun()


def display_mindmap():
    st.subheader("💡 Advanced Mindmap (Text-Based)")
    # (Mindmap code from previous example can be used here - ensure session_state keys are unique if needed)
    if 'mindmap_nodes' not in st.session_state:
        st.session_state.mindmap_nodes = {"root": {"children": {}}}

    def display_node(node_name, node_data, indent_level=0, path="root"):
        indent = "  " * indent_level
        st.text(f"{indent}- {node_name} (path: {path})")
        for child_name, child_data in node_data.get("children", {}).items():
            display_node(child_name, child_data, indent_level + 1, path=f"{path}/{child_name}")

    st.write("**Current Mindmap:**")
    display_node("root", st.session_state.mindmap_nodes["root"])

    parent_node_path = st.text_input("Parent Node Path (e.g., 'root' or 'root/concept1'):", "root", key="mm_parent")
    new_node_name = st.text_input("New Concept/Idea:", key="mm_new_node")

    if st.button("Add to Mindmap", key="mm_add"):
        if new_node_name:
            current_dict = st.session_state.mindmap_nodes
            path_parts = [p for p in parent_node_path.split('/') if p] # e.g. ['root', 'concept1']

            target_parent_dict = current_dict
            valid_path = True
            for part in path_parts:
                if part in target_parent_dict and "children" in target_parent_dict[part] : # For 'root' case
                     target_parent_dict = target_parent_dict[part]["children"]
                elif part in target_parent_dict: # For children directly
                    target_parent_dict = target_parent_dict[part]
                    if "children" not in target_parent_dict: # Ensure child dict exists
                         target_parent_dict["children"] = {}

                else: # For children not directly under current part, but under its 'children'
                    is_found = False
                    if "children" in target_parent_dict:
                        if part in target_parent_dict["children"]:
                            target_parent_dict = target_parent_dict["children"][part]
                            if "children" not in target_parent_dict: # Ensure child dict exists
                                target_parent_dict["children"] = {}
                            is_found = True
                    if not is_found:
                        st.error(f"Parent path part '{part}' not found in '{parent_node_path}'. Node not added.")
                        valid_path = False
                        break
            
            if valid_path:
                 if "children" not in target_parent_dict: # if the target_parent_dict itself should have children property
                      target_parent_dict["children"] = {}

                 if new_node_name not in target_parent_dict["children"]:
                    target_parent_dict["children"][new_node_name] = {"children": {}}
                    st.success(f"Added '{new_node_name}' under '{parent_node_path}'.")
                    st.rerun()
                 else:
                    st.warning(f"'{new_node_name}' already exists under '{parent_node_path}'.")
        else:
            st.warning("New concept cannot be empty.")

    if st.button("Clear Mindmap", key="mm_clear"):
        st.session_state.mindmap_nodes = {"root": {"children": {}}}
        st.rerun()


def display_text_skimming(text):
    st.subheader("✂️ Text Skimming (First & Last Sentences)")
    if text:
        sentences = text.split('.') # Simple sentence split
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            skimmed_text = sentences[0] + ". [...] " + sentences[-1] + "."
            st.write("**Skimmed Version:**")
            st.info(skimmed_text)
        elif sentences:
            st.write("**Skimmed Version (Only one sentence found):**")
            st.info(sentences[0] + ".")
        else:
            st.warning("Not enough sentences to skim.")
    else:
        st.info("Upload or enter notes to use text skimming.")

# --- Main Application ---
st.set_page_config(layout="wide", page_title="Gemini Advanced Study Assistant")
st.title("🎓 Gemini Powered Advanced Exam Study Assistant")

# --- API Key Input & YOLO Model Loading ---
st.sidebar.header("🔑 API & Model Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="api_key_input_main")
yolo_model_name = st.sidebar.selectbox(
    "Select YOLO Model",
    ("diagram_det.pt", "None (Disable Diagram Detection)"), # Add more if needed, ensure they are downloadable by ultralytics
    index=0, key="yolo_model_select"
)
st.session_state.model_name=yolo_model_name
# Add options for diagram detection heuristic
yolo_conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 0.9, 0.25, 0.05, key="yolo_conf")
yolo_min_objs = st.sidebar.slider("Min Objects to Flag as Diagram", 1, 10, 3, 1, key="yolo_min_obj")


api_configured_success = False
if api_key:
    api_configured_success = configure_api(api_key)
    if api_configured_success:
        st.sidebar.success("Gemini API Key Configured!")
    else:
        st.sidebar.error("Failed to configure Gemini API.")
else:
    st.sidebar.info("Please enter your Gemini API Key.")

yolo_model_instance = None
if api_configured_success: # Only load YOLO if API is fine (or make independent)
    if st.session_state.model_name != "None (Disable Diagram Detection)":
        with st.spinner(f"Loading YOLO model '{st.session_state.model_name}'... This may take time on first run."):
            yolo_model_instance = load_yolo_model(st.session_state.model_name)
        if yolo_model_instance:
            st.sidebar.success(f"YOLO model '{st.session_state.model_name}' loaded.")
        else:
            st.sidebar.error(f"Could not load YOLO model '{st.session_state.model_name}'. Diagram detection will be impaired.")
    else:
        st.sidebar.info("YOLO-based diagram detection is disabled.")


# --- Main Tabs ---
if api_configured_success: # Core AI features depend on this
    display_pomodoro()
    display_mindmap()

    tab1, tab2, tab3 = st.tabs(["📝 Notes & Diagrams", "📜 Question Papers Chat", "🧠 Concept Learning"])

    with tab1:
        st.header("📝 Notes Management & Diagram Analysis")

        uploaded_notes_file = st.file_uploader("Upload notes (TXT or PDF)", type=["txt", "pdf"], key="notes_upload_main")

        if 'main_notes_text' not in st.session_state:
            st.session_state.main_notes_text = None
        if 'main_processed_diagrams' not in st.session_state: # Store {image_pil, ocr_text, filename, yolo_detected_objects, is_potential_diagram, yolo_detections_summary}
            st.session_state.main_processed_diagrams = []
        if 'current_notes_filename' not in st.session_state:
            st.session_state.current_notes_filename = None


        if uploaded_notes_file is not None:
            # Use a button to trigger processing to give user control and allow re-processing
            if st.button("Process Uploaded Notes File", key="proc_notes_btn"):
                st.session_state.main_notes_text = None # Reset
                st.session_state.main_processed_diagrams = [] # Reset
                st.session_state.current_notes_filename = uploaded_notes_file.name

                file_content = uploaded_notes_file.getvalue() # Get bytes content once

                if uploaded_notes_file.type == "text/plain":
                    st.session_state.main_notes_text = load_text_data(file_content, uploaded_notes_file.name)
                    st.success("Text notes loaded!")
                elif uploaded_notes_file.type == "application/pdf":
                    pdf_bytes = load_pdf_data(file_content, uploaded_notes_file.name)
                    if pdf_bytes:
                        try:
                            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            full_text = "".join([page.get_text() for page in doc])
                            doc.close()
                            st.session_state.main_notes_text = full_text
                            st.success("PDF text content extracted!")
                        except Exception as e:
                            st.error(f"Error extracting text from PDF: {e}")
                            st.session_state.main_notes_text = "Error extracting text."

                        # Diagram extraction and YOLO processing
                        if yolo_model_instance:
                            with st.spinner("Extracting and analyzing images for diagrams (YOLO)..."):
                                st.session_state.main_processed_diagrams = extract_and_process_images_from_pdf(
                                    pdf_bytes,
                                    uploaded_notes_file.name,
                                    yolo_model_instance,
                                    ocr_enabled=True,
                                    yolo_confidence_threshold=yolo_conf_thresh,
                                    yolo_min_objects_for_diagram=yolo_min_objs
                                )
                            num_potential_diagrams = sum(1 for d in st.session_state.main_processed_diagrams if d["is_potential_diagram"])
                            st.success(f"Processed {len(st.session_state.main_processed_diagrams)} images. Flagged {num_potential_diagrams} as potential diagrams using YOLO.")
                        else:
                            st.info("YOLO model not available. Skipping diagram detection and OCR for images.")
                st.rerun()

        if st.session_state.main_notes_text:
            st.subheader("Your Notes Content:")
            with st.expander("View Notes", expanded=False):
                st.text_area("Notes", st.session_state.main_notes_text, height=200, disabled=True, key="notes_text_area")
            display_text_skimming(st.session_state.main_notes_text)

            st.subheader("Ask Questions About Your Notes:")
            notes_q = st.text_input("Enter your question:", key="notes_q_main")
            if st.button("Get Answer from Notes", key="ask_notes_main"):
                if notes_q:
                    prompt = f"Based on the following notes:\n\n{st.session_state.main_notes_text}\n\nAnswer the question: {notes_q}"
                    with st.spinner("Gemini is thinking..."):
                        answer = get_gemini_response(prompt)
                    st.markdown("**Answer:**")
                    st.info(answer if answer else "Sorry, I couldn't get an answer.")

        if st.session_state.main_processed_diagrams:
            st.subheader("🖼️ Extracted Images & Potential Diagrams (via YOLO & OCR)")
            potential_diagrams_found = False
            for i, data in enumerate(st.session_state.main_processed_diagrams):
                if data["is_potential_diagram"]:
                    potential_diagrams_found = True
                    st.markdown(f"--- \n **Potential Diagram {i+1} (from page {data['page']})**")
                    st.image(data["image_pil"], caption=f"Page {data['page']} - YOLO Objects: {data['yolo_detected_objects']} ({', '.join(data['yolo_detections_summary']) if data['yolo_detections_summary'] else 'None'}). OCR: '{data['ocr_text'][:100]}...'")
                    if st.button(f"Explain this Diagram (ID {i+1})", key=f"explain_diag_main_{i}"):
                        prompt_explain = ""
                        if data["ocr_text"] and not data["ocr_text"].startswith("OCR Error") and not data["ocr_text"].startswith("OCR not performed"):
                            prompt_explain = f"The following text was extracted using OCR from an image flagged as a potential diagram: '{data['ocr_text']}'. The image also had {data['yolo_detected_objects']} objects detected by YOLO (classes: {', '.join(data['yolo_detections_summary'])}). Explain this diagram or the concepts it might represent. If OCR text is minimal or unclear, focus on what such a diagram might mean given these clues."
                        else:
                            prompt_explain = f"An image was extracted from a PDF (page {data['page']}) and flagged as a potential diagram because it had {data['yolo_detected_objects']} objects detected by YOLO (classes: {', '.join(data['yolo_detections_summary'])}). OCR was unable to extract meaningful text or was not performed. Based on common diagram types containing multiple objects, what might this diagram represent? Explain generally."

                        with st.spinner("Gemini is analyzing the diagram info..."):
                            explanation = get_gemini_response(prompt_explain)
                        st.markdown(f"**Explanation for Diagram {i+1}:**")
                        st.info(explanation if explanation else "Sorry, I couldn't get an explanation.")
            if not potential_diagrams_found and any(st.session_state.main_processed_diagrams):
                st.markdown("No images were flagged as potential diagrams based on current YOLO settings (Min Objects, Confidence). Some images might have been extracted but not processed further.")
            elif not st.session_state.main_processed_diagrams and st.session_state.current_notes_filename and st.session_state.current_notes_filename.endswith(".pdf"):
                 st.info("No images found in the PDF or diagram processing was skipped.")


        # Downloadable data for NN
        if st.session_state.main_notes_text or st.session_state.main_processed_diagrams:
            st.subheader("⬇️ Download Processed Data for Future Use")
            nn_data_list = []
            if st.session_state.main_notes_text:
                nn_data_list.append({
                    "type": "full_notes_text",
                    "source_file": st.session_state.current_notes_filename or "unknown",
                    "content": st.session_state.main_notes_text,
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            for diag_data in st.session_state.main_processed_diagrams:
                if diag_data["is_potential_diagram"]: # Only save confirmed diagram data
                    nn_data_list.append({
                        "type": "diagram_ocr_text",
                        "source_file": diag_data["filename"],
                        "page": diag_data["page"],
                        "ocr_content": diag_data["ocr_text"],
                        "yolo_object_count": diag_data["yolo_detected_objects"],
                        "yolo_classes": diag_data["yolo_detections_summary"],
                        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        # Note: Image data itself isn't added to JSON for brevity,
                        # but you could save images separately and reference them.
                    })
            if nn_data_list:
                json_data_to_download = json.dumps(nn_data_list, indent=4)
                st.download_button(
                    label="Download Data for NN (JSON)",
                    data=json_data_to_download,
                    file_name=f"nn_study_data_{st.session_state.current_notes_filename.split('.')[0] if st.session_state.current_notes_filename else 'export'}_{time.strftime('%Y%m%d')}.json",
                    mime="application/json",
                    key="download_nn_data"
                )

    with tab2:
        st.header("📜 Question Papers Chat")
        uploaded_qp_file = st.file_uploader("Upload Question Paper (TXT or PDF)", type=["txt", "pdf"], key="qp_upload_main")

        if 'main_qp_text' not in st.session_state:
            st.session_state.main_qp_text = None
        if 'current_qp_filename' not in st.session_state:
            st.session_state.current_qp_filename = None


        if uploaded_qp_file is not None:
            if st.button("Process Question Paper", key="proc_qp_btn"):
                st.session_state.main_qp_text = None # Reset
                st.session_state.current_qp_filename = uploaded_qp_file.name
                qp_file_content = uploaded_qp_file.getvalue()

                if uploaded_qp_file.type == "text/plain":
                    st.session_state.main_qp_text = load_text_data(qp_file_content, uploaded_qp_file.name)
                elif uploaded_qp_file.type == "application/pdf":
                    pdf_bytes_qp = load_pdf_data(qp_file_content, uploaded_qp_file.name)
                    if pdf_bytes_qp:
                        try:
                            doc_qp = fitz.open(stream=pdf_bytes_qp, filetype="pdf")
                            st.session_state.main_qp_text = "".join([page.get_text() for page in doc_qp])
                            doc_qp.close()
                        except Exception as e:
                            st.error(f"Error extracting PDF text: {e}")
                            st.session_state.main_qp_text = "Error."
                if st.session_state.main_qp_text and st.session_state.main_qp_text != "Error.":
                    st.success("Question paper processed!")
                else:
                    st.error("Could not process the question paper.")
                st.rerun()

        if st.session_state.main_qp_text:
            st.subheader("Question Paper Content:")
            with st.expander("View Paper", expanded=False):
                st.text_area("QP", st.session_state.main_qp_text, height=300, disabled=True, key="qp_text_area")

            st.subheader("Chat about the Question Paper:")
            if "main_qp_messages" not in st.session_state:
                st.session_state.main_qp_messages = []

            for message in st.session_state.main_qp_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_qp_query = st.chat_input("Ask about the paper...")
            if user_qp_query:
                st.session_state.main_qp_messages.append({"role": "user", "content": user_qp_query})
                with st.chat_message("user"): st.markdown(user_qp_query)

                with st.spinner("Gemini is drafting response..."):
                    prompt = f"Context from Question Paper:\n\n{st.session_state.main_qp_text}\n\nUser's Question: {user_qp_query}"
                    response = get_gemini_response(prompt)
                    with st.chat_message("assistant"): st.markdown(response or "No response.")
                    st.session_state.main_qp_messages.append({"role": "assistant", "content": response or "Error."})
                    # No st.rerun() for chat_input usually needed

    with tab3:
        st.header("🧠 Concept Learning Aids")
        st.subheader("Generate Mnemonics")
        concept_mnem = st.text_area("Concept for Mnemonic:", height=100, key="mnem_concept_main")
        if st.button("Get Mnemonic", key="gen_mnem_main"):
            if concept_mnem:
                prompt = f"Generate creative mnemonics for: '{concept_mnem}'"
                with st.spinner("Thinking..."): mnemonics = get_gemini_response(prompt)
                st.info(mnemonics or "Could not generate.")
        st.subheader("Simplify Complex Topic")
        topic_simplify = st.text_area("Topic to Simplify:", height=100, key="simp_topic_main")
        if st.button("Simplify", key="simp_btn_main"):
            if topic_simplify:
                prompt = f"Explain simply: '{topic_simplify}'"
                with st.spinner("Thinking..."): explanation = get_gemini_response(prompt)
                st.success(explanation or "Could not explain.")

else:
    st.warning("Please enter your Gemini API Key in the sidebar to use the core AI features.")

st.sidebar.markdown("---")
st.sidebar.info("Advanced Study Assistant using Gemini & YOLO.")
