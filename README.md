# AR-Drawing

A Gesture Based Drawing Tool that uses hand gestures to create digital artwork in real-time. The application leverages computer vision and machine learning to interpret hand gestures and translate them into drawing actions.

## Features

- Real-time hand gesture recognition for drawing control
- Multiple gesture commands for different drawing actions
- Color picker and brush size adjustment
- Erase mode for easy corrections
- Save and load drawing capabilities
- Web interface through Streamlit

## Prerequisites

- Python 3.9 or higher
- Webcam
- Virtual environment (recommended)

## Installation

Install required packages: 
-- If you are using non-M1 chip computer, you might need to change the requirements to use regular tensorflow instead of tensorflow metal
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```

2. The application will open in your default web browser and access your webcam.

### Gesture Controls

- 👆 **Point**: Draw, select, and erase (when in erase mode)
- 👍 **Size up**: Increase brush size
- 👎 **Size down**: Decrease brush size
- 🤚 **Erase**: Activate erase mode
- 🅒 **Color**: Activate color picker

### Keyboard Controls

- `c`: Clear temporary drawing
- `x`: Clear ALL drawings
- `p`: Toggle persistent drawing visibility
- `m`: Switch between overlay and whiteboard modes
- `s`: Save your drawing

## Project Structure

- `gesture_drawing.py`: Main application logic
- `gesture_recognition.py`: Hand gesture recognition implementation
- `streamlit_app.py`: Web interface
- `hand_data_collection.py`: Data collection for training
- `keypoint_train.ipynb`: Model training notebook
- `images/`: Directory for saved drawings

### Directory Structure
```
AR-Drawing/
├── Gesture_Drawing_Report_files/
├── images/
├── gesture_drawing.py -- App that uses the trained model to draw with web cam
├── gesture_recognition.py -- Testing demo to recognise different gestures
├── gesture_recognition_1d.py
├── hand_data_collection.py -- App that collects hand data
├── keypoint_train.ipynb
├── keypoint_train_1d.py
├── streamlit_app.py
├── Gesture_Drawing_Report.html
├── Gesture_Drawing_Report.qmd
└── README.md
```

