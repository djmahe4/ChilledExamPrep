# Study_App_Home.py
import streamlit as st
from utils import configure_api_key, load_yolo_model # Import necessary functions from utils

st.set_page_config(
    page_title="ChilledExamPrep - Home",
    page_icon="🏠",
    layout="wide"
)

st.title("🎓 Gemini Powered Study Hub")
st.caption("Welcome! Configure your API key and use the tools from the sidebar.")

# --- Initialize Session State Variables ---
# API Key configuration status
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# YOLO Model instance
if "yolo_model_instance" not in st.session_state:
    st.session_state.yolo_model_instance = None
if "yolo_model_loaded" not in st.session_state:
    st.session_state.yolo_model_loaded = False


# --- Sidebar for API Key Configuration ---
st.sidebar.header("🔑 API Configuration")
api_key_input = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    value=st.session_state.gemini_api_key, # Persist input value across reruns if needed
    key="api_key_widget" # Unique key for the widget
)

if api_key_input and api_key_input != st.session_state.gemini_api_key:
    st.session_state.gemini_api_key = api_key_input
    # Attempt to configure API only if key changes or not configured
    with st.spinner("Configuring API key..."):
        configured = configure_api_key(st.session_state.gemini_api_key)
        if configured:
            st.session_state.api_key_configured = True
            st.sidebar.success("Gemini API Key Configured!")
        else:
            st.session_state.api_key_configured = False
            st.sidebar.error("Failed to configure Gemini API. Check the key.")
    st.rerun() # Rerun to update status immediately

elif st.session_state.api_key_configured:
    st.sidebar.success("Gemini API Key is configured.")
else:
    st.sidebar.info("Please enter your Gemini API Key to enable AI features.")


# --- Load YOLO Model (once and cache) ---
if not st.session_state.yolo_model_loaded:
    with st.spinner("Loading Diagram Detection Model... This may take time on first run."):
        model_instance = load_yolo_model() # This is cached by @st.cache_resource in utils
        if model_instance:
            st.session_state.yolo_model_instance = model_instance
            st.session_state.yolo_model_loaded = True
            # No sidebar message here, utils.load_yolo_model shows messages
        else:
            st.sidebar.error("Diagram detection model could not be loaded. Diagram features will be affected.")
            st.session_state.yolo_model_instance = None # Ensure it's None if loading failed
            st.session_state.yolo_model_loaded = True # Mark as attempted
elif st.session_state.yolo_model_instance:
    # st.sidebar.info("Diagram detection model is loaded.") # Optional: can be too verbose
    pass


# --- Global Tools on Home Page (Pomodoro, Mindmap) ---
if st.session_state.api_key_configured: # Only show if API is set, or make independent
    st.markdown("---")
    st.header("🛠️ Productivity Tools")
    col1, col2 = st.columns(2)

    with col1:
        # Pomodoro Timer (Simplified example, needs session state for its own values)
        st.subheader("🍅 Pomodoro Timer")
        if 'pomo_timer_active' not in st.session_state: st.session_state.pomo_timer_active = False
        if 'pomo_seconds' not in st.session_state: st.session_state.pomo_seconds = 25 * 60
        if 'pomo_is_break' not in st.session_state: st.session_state.pomo_is_break = False

        if st.session_state.pomo_is_break:
            timer_label = "Break Time Remaining"
            button_label = "Start Work (25 min)" if not st.session_state.pomo_timer_active else "Pause Break"
            duration_key = 5 * 60
        else:
            timer_label = "Work Time Remaining"
            button_label = "Start Pomodoro (25 min)" if not st.session_state.pomo_timer_active else "Pause Work"
            duration_key = 25 * 60

        if st.button(button_label, key="pomo_button"):
            st.session_state.pomo_timer_active = not st.session_state.pomo_timer_active
            if st.session_state.pomo_timer_active and st.session_state.pomo_seconds == 0 : # starting fresh
                 st.session_state.pomo_seconds = duration_key
            # If pausing, current time is preserved. If switching mode, time resets.

        if st.session_state.pomo_timer_active:
            mins, secs = divmod(st.session_state.pomo_seconds, 60)
            st.metric(timer_label, f"{mins:02d}:{secs:02d}")
            # This is a simplified display. A real timer needs st.experimental_rerun or JS.
            if st.button("Tick 1s", key="pomo_tick"): # Manual tick for demo
                if st.session_state.pomo_seconds > 0:
                    st.session_state.pomo_seconds -= 1
                else: # Timer ended
                    st.session_state.pomo_timer_active = False
                    st.session_state.pomo_is_break = not st.session_state.pomo_is_break
                    st.session_state.pomo_seconds = (5*60) if st.session_state.pomo_is_break else (25*60)
                    st.success("Time's up! Switching mode." if st.session_state.pomo_is_break else "Break's over! Back to work.")
                st.rerun()
        elif st.session_state.pomo_seconds == 0 and not st.session_state.pomo_timer_active: # Ready to start
             st.info(f"Ready for {'Break' if st.session_state.pomo_is_break else 'Work'}.")


    with col2:
        # Mindmap (Simplified example, needs its own session state)
        st.subheader("💡 Mindmap (Text-Based)")
        if 'mindmap_nodes' not in st.session_state: st.session_state.mindmap_nodes = {"root": {"children": {}}}
        # ... (Simplified Mindmap UI - refer to previous complete examples for full logic) ...
        def display_mm_node(name, data, level=0):
            st.text("  " * level + f"- {name}")
            for child_name, child_data in data.get("children", {}).items():
                display_mm_node(child_name, child_data, level + 1)
        display_mm_node("root", st.session_state.mindmap_nodes["root"])
        # Add node UI
        mm_parent = st.text_input("Parent Node (e.g. root)", "root", key="mm_parent_home")
        mm_new = st.text_input("New Concept", key="mm_new_home")
        if st.button("Add to Mindmap", key="mm_add_home"):
            # Simplified add logic, assumes parent exists directly under root for this example
            if mm_new and mm_parent == "root": # Highly simplified
                st.session_state.mindmap_nodes["root"]["children"][mm_new] = {"children": {}}
                st.success(f"Added '{mm_new}' to mindmap.")
                st.rerun()
            elif mm_new:
                 st.warning("Simplified: only adding to root supported in this demo.")

st.markdown("--- \nNavigate to other pages using the sidebar for more study tools!")
