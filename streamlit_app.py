import streamlit as st
import subprocess
import sys
import os
import threading
import atexit

st.set_page_config(
    page_title="AR Gesture Drawing",
    layout="wide"
)

# Title and description
st.title("AR Gesture Drawing")
st.markdown("""
This app launches the gesture drawing application directly from your webcam.

### Gestures:
- 👆 **Point**: To draw, select, and erase (when in erase mode)
- 👍 **Size up**: Thumbs up to increase brush size
- 👎 **Size down**: Thumbs down to decrease brush size
- 🤚 **Erase**: Flat hand to activate erase mode
- 🅒 **Color**: C-shaped hand motion to activate color picker

### Keyboard Controls:
- `c`: Clear temporary drawing
- `x`: Clear ALL drawings
- `p`: Toggle persistent drawing visibility
- `m`: Switch between overlay and whiteboard modes
- `s`: Save your drawing
- `ESC`: Exit application
""")

# Check if gesture_drawing.py exists
if not os.path.exists("gesture_drawing.py"):
    st.error("Error: gesture_drawing.py file not found in the current directory.")
    st.stop()

# Function to run the gesture drawing app in a separate process
def run_gesture_drawing():
    try:
        # Run the gesture_drawing.py script
        process = subprocess.Popen([sys.executable, "gesture_drawing.py"], 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        
        # Store the process in session state so we can terminate it later
        st.session_state.process = process
        
        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                st.session_state.output_area.text(line.strip())
        
        # Check for errors
        for line in iter(process.stderr.readline, ''):
            if line:
                st.session_state.error_area.error(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        # Set process to None when done
        st.session_state.process = None
        st.session_state.running = False
        
    except Exception as e:
        st.error(f"Error executing gesture_drawing.py: {e}")
        st.session_state.running = False

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
    st.session_state.process = None

# Create placeholder for output
if 'output_area' not in st.session_state:
    st.session_state.output_area = st.empty()
if 'error_area' not in st.session_state:
    st.session_state.error_area = st.empty()

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if not st.session_state.running:
        if st.button("Start Gesture Drawing"):
            st.session_state.running = True
            # Start the process in a background thread
            thread = threading.Thread(target=run_gesture_drawing)
            thread.daemon = True
            thread.start()
            st.success("Gesture drawing application started! A new window should open.")

# Additional information
if st.session_state.running:
    st.info("""
    **The gesture drawing application is running in a separate window.** 
    
    You can interact with it using your webcam and the gestures/keys described above.
    """)
    
    st.warning("""
    **Note:** Closing this Streamlit app will also close the gesture drawing application.
    Press ESC in the drawing window to exit.
    """)

# Add a cleanup handler to ensure the process is terminated when Streamlit exits
def cleanup():
    if 'process' in st.session_state and st.session_state.process:
        try:
            st.session_state.process.terminate()
        except:
            pass

atexit.register(cleanup)
