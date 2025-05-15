# utils.py
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import json
import time
import functools
#import cv2
import os
os.popen("pip uninstall opencv-python &&
pip install opencv-python-headless")
from ultralytics import YOLO


# --- Gemini API Configuration and Interaction ---

@st.cache_resource
def configure_api_key(api_key):
    """Configures the Gemini API key."""
    try:
        if api_key:
            genai.configure(api_key=api_key)
            return True
        return False
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return False

def retry(retries=3, delay=5, backoff=2):
    """Retry decorator for resilient API calls."""
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
                        st.error(f"API Error after multiple retries: {e}")
                        if "API_KEY_INVALID" in str(e) or "PERMISSION_DENIED" in str(e) or "API key not valid" in str(e):
                            st.error("Critical API Error: Please ensure your Gemini API key is correct.")
                        return None
                    st.warning(f"API attempt failed. Retrying in {_delay}s... (Retries left: {_retries}). Error: {e}")
                    time.sleep(_delay)
                    _delay *= backoff
            return None
        return wrapper
    return decorator

@retry(retries=3, delay=5)
def get_gemini_response(prompt_text, model_name="gemini-1.5-flash-latest"):
    """Gets response from Gemini API with retry logic."""
    if not st.session_state.get("api_key_configured", False):
        st.error("Gemini API key not configured. Please set it on the Home page.")
        return "Error: API Key not configured."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        # Specific error handling might be done by the retry decorator
        st.error(f"Gemini API Error during generation: {e}")
        return None

# --- Data Loading and Processing Functions (Cached) ---

@st.cache_data
def load_text_data(uploaded_file_content, _filename_key):
    """Loads text data from uploaded file content."""
    if uploaded_file_content is not None:
        return uploaded_file_content.decode()
    return None

@st.cache_data
def load_pdf_data(uploaded_file_content, _filename_key):
    """Loads PDF data from uploaded file content."""
    if uploaded_file_content is not None:
        return uploaded_file_content # Return bytes
    return None

@st.cache_resource # Caches the model itself
def load_yolo_model():
    """Loads the pre-defined YOLO model."""
    # User wants to use 'diagram_det.pt'
    custom_model_path = 'diagram_det.pt'
    fallback_model = 'yolov8n.pt' # A general, small model

    chosen_model_path = custom_model_path

    # Check if the custom model file exists. If not, use fallback and warn.
    # This assumes 'diagram_det.pt' is in the same directory as the script or a resolvable path.
    if not os.path.exists(custom_model_path):
        st.warning(
            f"Custom YOLO model '{custom_model_path}' not found. "
            f"Falling back to '{fallback_model}'. Place '{custom_model_path}' "
            f"in the project root for diagram-specific detection."
        )
        chosen_model_path = fallback_model
    
    try:
        model = YOLO(chosen_model_path)
        st.success(f"YOLO model '{chosen_model_path}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model '{chosen_model_path}': {e}. Ensure Ultralytics is installed and model path is correct.")
        # If fallback also fails, this will return None
        if chosen_model_path == fallback_model and chosen_model_path != custom_model_path : # If already tried fallback
             st.error(f"Fallback YOLO model '{fallback_model}' also failed to load.")
        return None


# Note: The _yolo_model_object_id is added to the cache key for extract_and_process_images_from_pdf
# to ensure that if somehow a different model object was used (e.g., if we allowed model switching again),
# the cache would reflect that. Since the model is fixed and loaded once with @st.cache_resource,
# its id() will be constant for a given app run/session where the resource is cached.
@st.cache_data
def extract_and_process_images_from_pdf(pdf_bytes, _pdf_filename_key, _yolo_model_object_id, yolo_confidence_threshold=0.25, yolo_min_objects_for_diagram=3):
    """
    Extracts images from PDF, uses a pre-loaded YOLO model for object detection,
    and performs OCR on images flagged as potential diagrams.
    The yolo_model object itself is not passed here, but its ID is used for cache invalidation.
    The actual model is retrieved from st.session_state.
    """
    processed_images_data = []
    yolo_model = st.session_state.get("yolo_model_instance") # Retrieve the loaded model

    if not yolo_model:
        st.warning("YOLO model not available in session state for image processing.")
        return processed_images_data
    if not pdf_bytes:
        return processed_images_data

    try:
        pdf_document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") # Ensure using io.BytesIO for bytes
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image_data = pdf_document.extract_image(xref)
                image_bytes = base_image_data["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                yolo_results = yolo_model(pil_image, conf=yolo_confidence_threshold, verbose=False)
                detected_objects_count = len(yolo_results[0].boxes)
                is_potential_diagram = detected_objects_count >= yolo_min_objects_for_diagram

                ocr_text = "OCR not performed."
                if is_potential_diagram:
                    try:
                        ocr_text = pytesseract.image_to_string(pil_image)
                    except Exception as ocr_err:
                        ocr_text = f"OCR Error: {ocr_err}"
                
                processed_images_data.append({
                    "page": page_num + 1,
                    "image_pil": pil_image,
                    "ocr_text": ocr_text,
                    "filename": _pdf_filename_key, # Use the passed filename key
                    "yolo_detected_objects": detected_objects_count,
                    "is_potential_diagram": is_potential_diagram,
                    "yolo_detections_summary": [yolo_model.names[int(cls)] for cls in yolo_results[0].boxes.cls] if yolo_results and yolo_results[0].boxes else []
                })
        pdf_document.close()
    except Exception as e:
        st.error(f"Error processing PDF for images/YOLO/OCR: {e}")
    return processed_images_data
