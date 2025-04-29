import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# Load the CNN1D model 
model_path = 'model/keypoint_classifier/keypoint_classifier_1d.keras'
model = tf.keras.models.load_model(model_path)
print(f"Loaded 1D CNN model from {model_path}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


label_file_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
if os.path.exists(label_file_path):
    with open(label_file_path, 'r') as f:
        labels = [line.strip() for line in f]
else:
    labels = ["size up", "size down", "nothing", "erase", "point", "color", "random"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas for drawing
ret, frame = cap.read()
if not ret:
    print("Failed to grab initial frame")
    exit()

# Create a white canvas instead of transparent
# White background
canvas = np.ones_like(frame) * 255  
# Transparent overlay for combined view
draw_overlay = np.zeros_like(frame) 

# Create a persistent canvas that never gets cleared unless explicitly requested
persistent_canvas = np.zeros_like(frame) 
# Create a mask to track where drawings have been made
drawing_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

# Drawing settings
# Red 
draw_color = (0, 0, 255)  
# Default thickness 
brush_thickness = 20 
# Minimum brush size
min_brush_thickness = 5 
# Maximum brush size
max_brush_thickness = 50 
# How much to increase/decrease by
brush_size_increment = 5 
# Fixed radius for the eraser tool
eraser_radius = 30 
is_drawing = False
last_point = None
# Store all points for debugging
draw_points = [] 
# For line smoothing
smoothing_points = [] 
# Reduced for more immediate response
smoothing_window = 2  

# Fingertip indices for all fingers 
# Thumb, index, middle, ring, pinky tips
FINGERTIP_INDICES = [4, 8, 12, 16, 20] 

# Brush size control
# Track when we last changed the brush size
last_size_change_time = 0 
# Cooldown in seconds between size changes
size_change_cooldown = 1.0  

# Flag to control whether to show the persistent drawing
show_persistent_drawing = True

# Canvas dimensions (Adjust based on camera resolution)
canvas_width = 640  
canvas_height = 480  

# Color picker settings
# Flag to track if color picker is active
color_picker_active = False 
color_options = [
    (0, 0, 255),    # Red
    (0, 127, 255),  # Orange 
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 127),  # Purple
    (255, 0, 255),  # Pink
    (0, 0, 0),      # Black
    (255, 255, 255) # White
]
color_names = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Black", "White"]
# Default to red
selected_color_idx = 0  

# Erase mode settings
erase_mode_active = False  

# Convert raw MediaPipe landmark coordinates to screen coordinates
def convert_landmark_to_screen_coordinates(landmark, image_shape):
    # MediaPipe provides normalized coordinates (0.0-1.0)
    # Convert to pixel coordinates based on image dimensions
    h, w = image_shape[:2]
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    # Ensure coordinates are within bounds
    x = max(0, min(w-1, x))
    y = max(0, min(h-1, y))
    return x, y

# Coordinate normalization function for displaying coordinates
def normalize_coordinates(x, y):
    # Ensure coordinates are positive and within canvas bounds
    x = max(0, min(canvas_width-1, x))
    y = max(0, min(canvas_height-1, y))
    return int(x), int(y)

# Initialize drawing paths 
drawing_paths = []
current_path = []

# Drawing mode
drawing_mode = "overlay" 

# Define index finger tip index
INDEX_FINGER_TIP = 8  

# Define confidence threshold for point gesture
POINT_CONFIDENCE_THRESHOLD = 0.6  


DEBUG_MODE = True

# Convert landmark coordinates to relative coordinates and normalize
def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for i, point in enumerate(temp_landmark_list):
        # Use the wrist as the base point
        if i == 0: 
            base_x, base_y = point[0], point[1]
        
        temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x
        temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y
    
    # Flatten the list for 1D CNN
    temp_landmark_list = list(np.array(temp_landmark_list).flatten())
    
    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value != 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    
    return temp_landmark_list

# Predict gesture using the 1D CNN model
def predict_gesture(landmark_list):
    # Prepare input data 
    input_data = np.array([landmark_list], dtype=np.float32)
    
    # Make prediction with the model
    predictions = model.predict(input_data, verbose=0)
    
    # Get the index of the highest probability
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class_idx, confidence

# Define a function to get smoothed point
def get_smoothed_point(new_point):
    global smoothing_points
    smoothing_points.append(new_point)
    if len(smoothing_points) > smoothing_window:
        smoothing_points.pop(0)
    
    # Average the points
    x_sum = sum(p[0] for p in smoothing_points)
    y_sum = sum(p[1] for p in smoothing_points)
    return (x_sum // len(smoothing_points), y_sum // len(smoothing_points))

# Function to draw the color wheel/palette
def draw_color_picker(image):
    h, w = image.shape[:2]
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (220, 220, 220), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw title
    cv2.putText(image, "COLOR SELECTION", (w//4 + 20, h//4 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw color options in a grid
    color_size = 50
    margin = 20
    cols = 3
    rows = (len(color_options) + cols - 1) // cols  # Ceiling division
    
    start_x = w//2 - ((cols * color_size) + (cols-1) * margin) // 2
    start_y = h//4 + 60
    
    # Draw each color option
    for i, color in enumerate(color_options):
        row = i // cols
        col = i % cols
        
        x = start_x + col * (color_size + margin)
        y = start_y + row * (color_size + margin)
        
        # Draw color square
        cv2.rectangle(image, (x, y), (x + color_size, y + color_size), color, -1)
        cv2.rectangle(image, (x, y), (x + color_size, y + color_size), (0, 0, 0), 2)
        
        # Highlight selected color
        if i == selected_color_idx:
            cv2.rectangle(image, (x-5, y-5), (x + color_size+5, y + color_size+5), (0, 255, 255), 3)
        
        # Add color name
        cv2.putText(image, color_names[i], (x, y + color_size + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Draw close button (X)
    close_x = 3*w//4 - 30
    close_y = h//4 + 30
    cv2.rectangle(image, (close_x - 15, close_y - 15), (close_x + 15, close_y + 15), (200, 200, 200), -1)
    cv2.rectangle(image, (close_x - 15, close_y - 15), (close_x + 15, close_y + 15), (0, 0, 0), 2)
    cv2.putText(image, "X", (close_x - 8, close_y + 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    return close_x, close_y, start_x, start_y, color_size, margin, cols

# Function to check if a point is within a specific color in the picker
def is_point_in_color(point, color_idx, start_x, start_y, color_size, margin, cols):
    row = color_idx // cols
    col = color_idx % cols
    
    x = start_x + col * (color_size + margin)
    y = start_y + row * (color_size + margin)
    
    return (x <= point[0] <= x + color_size and 
            y <= point[1] <= y + color_size)

# Function to check if a point is within the close button
def is_point_in_close_button(point, close_x, close_y):
    return (close_x - 15 <= point[0] <= close_x + 15 and 
            close_y - 15 <= point[1] <= close_y + 15)

# Function to display current brush size as a preview circle
def draw_brush_size_preview(image, thickness):
    # Position in top-right
    preview_pos = (image.shape[1] - 50, 50) 
    # Draw outer white circle
    cv2.circle(image, preview_pos, thickness//2 + 2, (255, 255, 255), 2)
    # Draw inner colored circle representing the brush
    cv2.circle(image, preview_pos, thickness//2, draw_color, -1)
    # Add text label
    cv2.putText(image, f"Size: {thickness}", (image.shape[1] - 120, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Add color name
    cv2.putText(image, f"Color: {color_names[selected_color_idx]}", (image.shape[1] - 150, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Function to handle brush size changes
def update_brush_size(gesture, current_time):
    """Update brush size based on gesture with cooldown"""
    global brush_thickness, last_size_change_time
    
    # Check if cooldown period has passed
    if current_time - last_size_change_time < size_change_cooldown:
        return False 
    
    changed = False
    if gesture == "size up" and brush_thickness < max_brush_thickness:
        brush_thickness += brush_size_increment
        changed = True
        print(f"Brush size increased to {brush_thickness}")
    elif gesture == "size down" and brush_thickness > min_brush_thickness:
        brush_thickness -= brush_size_increment
        changed = True
        print(f"Brush size decreased to {brush_thickness}")
    
    if changed:
        # Reset cooldown timer
        last_size_change_time = current_time  
        return True
    
    return False

# Function to erase drawings around a point - simplified and more direct
def erase_around_point(point, radius, canvas):
    # Simply draw a black (filled) circle directly on the canvas to erase
    cv2.circle(canvas, point, radius, (0, 0, 0), -1)
    # Also update the drawing mask to indicate this area has been modified
    cv2.circle(drawing_mask, point, radius, 0, -1)
    return True 

# Function to draw the erase mode interface
def draw_erase_mode_interface(image):
    h, w = image.shape[:2]
    
    # Draw indicator at top of screen
    cv2.rectangle(image, (0, 0), (w, 50), (50, 50, 50), -1)
    cv2.putText(image, "ERASE MODE ACTIVE", (w//2 - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw a larger and more visible exit button
    exit_button_x = w - 120
    exit_button_y = 25
    # Increase button size significantly
    button_width = 110  # Slightly wider to accommodate text
    button_height = 35
    
    # Draw button with red background for better visibility
    cv2.rectangle(image, 
                 (exit_button_x - button_width//2, exit_button_y - button_height//2), 
                 (exit_button_x + button_width//2, exit_button_y + button_height//2), 
                 (0, 0, 200), -1)  # Red background
    
    # Add a white border
    cv2.rectangle(image, 
                 (exit_button_x - button_width//2, exit_button_y - button_height//2), 
                 (exit_button_x + button_width//2, exit_button_y + button_height//2), 
                 (255, 255, 255), 2)  # White border
    
    # Get text size to properly center it
    text = "EXIT ERASE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65  # Slightly smaller font
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate text position to center it in the button
    text_x = exit_button_x - text_size[0] // 2
    text_y = exit_button_y + text_size[1] // 2
    
    # Add text with proper centering
    cv2.putText(image, text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)  # White text
    
    # Draw eraser size indicator
    cv2.putText(image, f"Eraser Size: {eraser_radius}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return exit_button_x, exit_button_y

# Function to check if a point is within the exit button - update to match new button size
def is_point_in_exit_button(point, exit_x, exit_y):
    button_width = 110  # Match the width from above
    button_height = 35
    return (exit_x - button_width//2 <= point[0] <= exit_x + button_width//2 and 
            exit_y - button_height//2 <= point[1] <= exit_y + button_height//2)

# Define a helper function to create a brighter version of a color for outlines
def get_outline_color(color):
    b, g, r = color 
    # Make each channel brighter by 50 or max out at 255
    # This ensures the outline is visible but matches the color theme
    bright_b = min(b + 50, 255)
    bright_g = min(g + 50, 255)
    bright_r = min(r + 50, 255)
    return (bright_b, bright_g, bright_r)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam")
        break
    
    # Update canvas dimensions based on actual frame size
    h, w = image.shape[:2]
    canvas_height, canvas_width = h, w
    
    # Flip horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe
    results = hands.process(image_rgb)
    
    # Variables to track if we're pointing this frame
    pointing_this_frame = False
    current_point = None
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw MediaPipe hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract landmark coordinates
            landmark_list = []
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                # Convert MediaPipe landmark coordinates to screen coordinates
                x, y = convert_landmark_to_screen_coordinates(landmark, image.shape)
                landmark_list.append([x, y])
            
            # Process landmarks for classification
            processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Predict gesture - no need to reshape for 1D model
            predicted_class_idx, confidence = predict_gesture(processed_landmark_list)
            if predicted_class_idx < len(labels):
                predicted_label = labels[predicted_class_idx]
                
                # Handle size up/down gestures with cooldown
                if predicted_label in ["size up", "size down"]:
                    current_time = time.time()
                    size_changed = update_brush_size(predicted_label, current_time)
                    if size_changed:
                        # Display size change notification
                        size_text = f"Brush size: {brush_thickness}"
                        cv2.putText(image, size_text, (10, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Handle color gesture
                elif predicted_label == "color" and confidence > 0.7:
                    # Activate color picker mode if not already active
                    if not color_picker_active:
                        color_picker_active = True
                        print("Color picker activated")
                
                # Handle erase gesture
                elif predicted_label == "erase" and confidence > 0.7:
                    # Activate erase mode if not already active
                    if not erase_mode_active and not color_picker_active:
                        erase_mode_active = True
                        print("Erase mode activated")
            else:
                predicted_label = f"Unknown ({predicted_class_idx})"
            
            # Display the gesture label with confidence
            info_text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(image, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Initialize current_point for color picker
            current_point = None
            
            # If color picker is active, handle the cursor
            if color_picker_active:
                # Get index finger tip position for a cursor
                index_finger_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
                tip_x, tip_y = convert_landmark_to_screen_coordinates(index_finger_landmark, image.shape)
                current_point = (tip_x, tip_y)
                
                # Draw cursor at fingertip for immediate feedback
                cv2.circle(image, current_point, 15, (0, 255, 255), -1)  
                cv2.circle(image, current_point, 15, (0, 0, 0), 2)    
                
                # Add a small crosshair for precise selection
                cv2.line(image, (tip_x-10, tip_y), (tip_x+10, tip_y), (0, 0, 0), 2)
                cv2.line(image, (tip_x, tip_y-10), (tip_x, tip_y+10), (0, 0, 0), 2)
                
                # Disable pointing_this_frame in color picker mode to prevent drawing
                pointing_this_frame = False
            
            # If "point" gesture is detected, highlight the index finger tip and draw
            if predicted_label == "point" and confidence > POINT_CONFIDENCE_THRESHOLD:
                # Get index finger tip landmark directly from MediaPipe
                index_finger_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
                
                # Convert to screen coordinates
                x, y = convert_landmark_to_screen_coordinates(index_finger_landmark, image.shape)
                
                # Set current point for drawing 
                current_point = (x, y)
                pointing_this_frame = True
                
                # Draw a more visible highlight at the index finger tip
                # Inner solid circle (red)
                cv2.circle(image, (x, y), 12, (0, 0, 255), -1)
                # Middle ring (white)
                cv2.circle(image, (x, y), 18, (255, 255, 255), 3)
                # Outer ring (blue)
                cv2.circle(image, (x, y), 24, (255, 0, 0), 2)
                
                # Display fingertip coordinates with more visibility
                tip_text = f"Position: ({x}, {y})"
                cv2.putText(image, tip_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                # If not pointing, reset drawing state
                is_drawing = False
                last_point = None
            
            # If erase mode is active, handle it
            if erase_mode_active and results.multi_hand_landmarks:
                # Skip drawing mode if erase mode is active
                if predicted_label == "point" and confidence > 0.7:
                    # Get index finger tip position for erasing
                    index_finger_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
                    tip_x, tip_y = convert_landmark_to_screen_coordinates(index_finger_landmark, image.shape)
                    current_point = (tip_x, tip_y)
                    
                    # Draw eraser cursor at fingertip position
                    cv2.circle(image, current_point, eraser_radius, (0, 255, 255), 3) 
                    cv2.circle(image, current_point, eraser_radius, (0, 0, 0), 1)   
                    
                    # Erase at the finger position from all canvases
                    erase_around_point(current_point, eraser_radius, persistent_canvas)
                    erase_around_point(current_point, eraser_radius, canvas)
                    erase_around_point(current_point, eraser_radius, draw_overlay)
                    
                    # Also erase directly on the image for immediate feedback
                    cv2.circle(image, current_point, eraser_radius, (0, 0, 0), -1)
                    
                    # Debug info
                    if DEBUG_MODE:
                        print(f"Erasing at position: {current_point}")
                
                # Skip normal drawing mode
                pointing_this_frame = False

    # Handle drawing logic
    if pointing_this_frame and not color_picker_active:
        if not is_drawing:
            # Start a new drawing stroke
            is_drawing = True
            last_point = current_point
            # Always start with the exact finger position (no smoothing for first point)
            smoothing_points = [current_point] 
            current_path = [current_point]
            
            # Draw a dot at the start point on all canvases, including the persistent one
            cv2.circle(image, current_point, brush_thickness//2, draw_color, -1)
            cv2.circle(canvas, current_point, brush_thickness//2, draw_color, -1)
            cv2.circle(draw_overlay, current_point, brush_thickness//2, draw_color, -1)
            cv2.circle(persistent_canvas, current_point, brush_thickness//2, draw_color, -1)
            # Update the drawing mask
            cv2.circle(drawing_mask, current_point, brush_thickness//2, 255, -1)
            
            # Debug message
            print(f"Starting to draw at {current_point}")
        elif last_point is not None:  
            smoothed_point = get_smoothed_point(current_point)
            
            if (abs(last_point[0] - smoothed_point[0]) > 1 or 
                abs(last_point[1] - smoothed_point[1]) > 1): 
                
                # Continue the stroke by drawing a line from last point to current
                # Draw directly on the image for immediate feedback
                outline_color = get_outline_color(draw_color)
                cv2.line(image, last_point, smoothed_point, outline_color, brush_thickness + 6, cv2.LINE_AA)  # Colored outline
                cv2.line(image, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)  # Color line
                
                # Add to the persistent canvases
                cv2.line(canvas, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)
                cv2.line(draw_overlay, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)
                
                # Add to the persistent canvas that never gets cleared - with colored outline
                cv2.line(persistent_canvas, last_point, smoothed_point, outline_color, brush_thickness + 6, cv2.LINE_AA)  # Colored outline
                cv2.line(persistent_canvas, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)  # Color line
                # Update the drawing mask
                cv2.line(drawing_mask, last_point, smoothed_point, 255, brush_thickness + 6, cv2.LINE_AA)
                
                # Store the point in both lists
                draw_points.append(smoothed_point)
                current_path.append(smoothed_point)
                
                # Update last point
                last_point = smoothed_point
                
                # Debug info
                if DEBUG_MODE:
                    print(f"Drawing line from {last_point} to {smoothed_point}")
        else:
            # Not pointing, finish the current stroke if we were drawing
            if is_drawing and current_path:
                drawing_paths.append(current_path)  # Store the completed path
                current_path = []
            
            # Reset drawing state
            is_drawing = False
            last_point = None
            smoothing_points = []  # Reset smoothing
    else:
        # Not pointing or in color picker mode, finish the current stroke if we were drawing
        if is_drawing and current_path:
            drawing_paths.append(current_path.copy())  # Store a COPY of the completed path
            current_path = []
        
        # Reset drawing state
        is_drawing = False
        last_point = None
        smoothing_points = []  
    
    # Update visualization to always include persistent drawings
    if drawing_mode == "overlay" and not erase_mode_active:
        # First create a blank overlay for finished paths
        visible_overlay = np.zeros_like(frame)
        
        # Draw all completed paths
        for path in drawing_paths:
            if len(path) > 1:
                for i in range(1, len(path)):
                    # Draw colored outline that matches the stroke color (but brighter)
                    outline_color = get_outline_color(draw_color)
                    cv2.line(visible_overlay, path[i-1], path[i], 
                            outline_color, brush_thickness + 6, cv2.LINE_AA)
                    # Then draw colored line on top
                    cv2.line(visible_overlay, path[i-1], path[i], 
                            draw_color, brush_thickness, cv2.LINE_AA)
        
        # Draw current path separately to ensure it's always visible
        if current_path and len(current_path) > 1 and not color_picker_active:
            for i in range(1, len(current_path)):
                # Draw on both the visible overlay AND directly on the image
                # Draw colored outline
                outline_color = get_outline_color(draw_color)
                cv2.line(visible_overlay, current_path[i-1], current_path[i], 
                        outline_color, brush_thickness + 6, cv2.LINE_AA)
                cv2.line(image, current_path[i-1], current_path[i], 
                        outline_color, brush_thickness + 6, cv2.LINE_AA)
                
                # Then draw colored line on top
                cv2.line(visible_overlay, current_path[i-1], current_path[i], 
                        draw_color, brush_thickness, cv2.LINE_AA)
                cv2.line(image, current_path[i-1], current_path[i], 
                        draw_color, brush_thickness, cv2.LINE_AA)
        
        # Combine with the camera image 
        combined_image = cv2.addWeighted(image, 1.0, visible_overlay, 1.0, 0)
        
        # Use the ACTUAL persistent canvas for overlay to show erasing in real-time
        if show_persistent_drawing:
            # Apply the persistent canvas directly
            # Use the drawing_mask instead of creating a mask from grayscale
            mask = drawing_mask > 0
            # Only apply pixels that are marked in the drawing mask
            combined_image[mask] = persistent_canvas[mask]
    elif drawing_mode == "whiteboard" and not erase_mode_active:
        # Redraw all paths on a fresh canvas each frame
        canvas_draw = np.ones_like(frame) * 255  # Fresh white canvas
        
        # Always add the persistent drawings
        if show_persistent_drawing:
            # Use the drawing_mask instead of creating a mask from grayscale
            mask = drawing_mask > 0
            canvas_draw[mask] = persistent_canvas[mask]
        
        # Draw current path if active
        if current_path and len(current_path) > 1:
            for i in range(1, len(current_path)):
                cv2.line(canvas_draw, current_path[i-1], current_path[i], 
                        draw_color, brush_thickness, cv2.LINE_AA)
        
        combined_image = canvas_draw.copy()
        
        # Add a small camera preview in the corner
        h, w = image.shape[:2]
        preview_size = (w // 4, h // 4)
        preview = cv2.resize(image, preview_size)
        combined_image[10:10+preview_size[1], 10:10+preview_size[0]] = preview

    # Draw indicator when actively drawing
    if pointing_this_frame and not color_picker_active:
        status_text = "DRAWING ACTIVE"
        cv2.putText(combined_image, status_text, (combined_image.shape[1] - 250, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # Add a visual indicator that drawing is happening
        cv2.circle(combined_image, (combined_image.shape[1] - 30, 30), 15, (0, 255, 0), -1)

    # Display UI controls
    cv2.putText(combined_image, "Press 'c' to clear temp drawing", (10, combined_image.shape[0] - 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 'x' to clear ALL drawings", (10, combined_image.shape[0] - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 'p' to toggle persistence", (10, combined_image.shape[0] - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 'm' to switch modes", (10, combined_image.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 's' to save drawing", (10, combined_image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Add coordinate system visualization to help understanding
    def draw_coordinate_system(image):
        """Draw coordinate system visualization on the image"""
        # Draw coordinate axes
        # X-axis - red
        cv2.line(image, (0, 20), (150, 20), (0, 0, 255), 2)
        cv2.arrowedLine(image, (130, 20), (150, 20), (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(image, "X+", (155, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Y-axis - green
        cv2.line(image, (20, 0), (20, 150), (0, 255, 0), 2)
        cv2.arrowedLine(image, (20, 130), (20, 150), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(image, "Y+", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Origin
        cv2.circle(image, (20, 20), 5, (255, 0, 255), -1)
        cv2.putText(image, "(0,0)", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Before showing the combined image, add coordinate system
    # Draw the coordinate system visualization in both modes
    if drawing_mode == "overlay":
        draw_coordinate_system(combined_image)
    else:
        draw_coordinate_system(combined_image)

    # Draw color picker if active
    if color_picker_active:
        # First, disable drawing mode while picker is active
        pointing_this_frame = False
        
        # Draw the picker interface
        close_x, close_y, start_x, start_y, color_size, margin, cols = draw_color_picker(combined_image)
        
        # Show color picker status indicator
        status_text = "COLOR PICKER ACTIVE"
        cv2.putText(combined_image, status_text, (combined_image.shape[1] - 250, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        # Add a visual indicator
        cv2.circle(combined_image, (combined_image.shape[1] - 30, 30), 15, (0, 255, 255), -1)
        
        # Get current point for cursor if it wasn't already set
        if current_point is None and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
                tip_x, tip_y = convert_landmark_to_screen_coordinates(index_finger_landmark, image.shape)
                current_point = (tip_x, tip_y)
        
        # If point gesture is active, check for selections
        if predicted_label == "point" and confidence > 0.7 and current_point is not None:
            # Check if close button clicked
            if is_point_in_close_button(current_point, close_x, close_y):
                color_picker_active = False
                print(f"Color picker closed. Selected color: {color_names[selected_color_idx]}")
            
            # Check if any color clicked
            for i in range(len(color_options)):
                if is_point_in_color(current_point, i, start_x, start_y, color_size, margin, cols):
                    selected_color_idx = i
                    draw_color = color_options[i]
                    print(f"Selected color: {color_names[i]}")
        
        # Always draw the cursor when the color picker is active and we have a point
        if current_point is not None:
            # Draw cursor at fingertip - make it more visible on the combined image
            cv2.circle(combined_image, current_point, 15, (0, 255, 255), -1)  # Yellow dot
            cv2.circle(combined_image, current_point, 15, (0, 0, 0), 2)       # Black outline
            
            # Add a small crosshair for precise selection
            tip_x, tip_y = current_point
            cv2.line(combined_image, (tip_x-10, tip_y), (tip_x+10, tip_y), (0, 0, 0), 2)
            cv2.line(combined_image, (tip_x, tip_y-10), (tip_x, tip_y+10), (0, 0, 0), 2)

    # Draw brush size preview
    draw_brush_size_preview(combined_image, brush_thickness)

    # After handling color picker but before showing the final image, update the erase mode handling:
    # Draw erase mode interface if active
    if erase_mode_active:
        # Disable drawing while in erase mode
        pointing_this_frame = False
        
        # First ensure we're using the most up-to-date canvas
        # This is critical - we need to make sure the combined_image shows the result of erasing
        if drawing_mode == "overlay":
            # Create a blank base
            base_image = image.copy()
            
            # Apply persistent canvas with erased regions properly showing up
            # Use the drawing_mask instead of creating a mask from grayscale
            mask = drawing_mask > 0
            base_image[mask] = persistent_canvas[mask]
            
            # Use this as our combined image
            combined_image = base_image.copy()
        else: 
            # Create a fresh white canvas
            whiteboard = np.ones_like(frame) * 255
            
            # Apply persistent canvas with erased regions
            # Use the drawing_mask instead of creating a mask from grayscale
            mask = drawing_mask > 0
            whiteboard[mask] = persistent_canvas[mask]
            
            # Use this for our combined image
            combined_image = whiteboard.copy()
            
            # Add a small camera preview in the corner
            h, w = image.shape[:2]
            preview_size = (w // 4, h // 4)
            preview = cv2.resize(image, preview_size)
            combined_image[10:10+preview_size[1], 10:10+preview_size[0]] = preview
        
        # Draw the erase mode interface
        exit_button_x, exit_button_y = draw_erase_mode_interface(combined_image)
        
        # If point gesture is active, check if exit button was clicked
        if predicted_label == "point" and confidence > 0.7 and current_point is not None:
            if is_point_in_exit_button(current_point, exit_button_x, exit_button_y):
                erase_mode_active = False
                print("Exited erase mode")
                
                # Clear existing drawing paths that might have been erased
                drawing_paths = []
                
                # Ensure the draw_overlay matches the persistent canvas for visual consistency
                draw_overlay = persistent_canvas.copy()
            
            # Draw cursor at current point for feedback
            cv2.circle(combined_image, current_point, 10, (0, 255, 255), -1)
            cv2.circle(combined_image, current_point, 10, (0, 0, 0), 2)

    # Show the combined image
    cv2.imshow('AR Drawing with Hand Gestures', combined_image)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  
        break
    elif key == ord('c'):  # Clear only the temporary drawing
        canvas = np.ones_like(frame) * 255  # White background
        draw_overlay = np.zeros_like(frame)  # Clear overlay
        draw_points = []  # Clear stored points
        drawing_paths = []  # Clear all paths
        current_path = []  # Clear current path
        print("Temporary canvas cleared (permanent drawing remains)")
    elif key == ord('x'):  # Clear ALL drawings including persistent ones
        canvas = np.ones_like(frame) * 255  # White background
        draw_overlay = np.zeros_like(frame)  # Clear overlay
        persistent_canvas = np.zeros_like(frame)  # Clear persistent canvas
        drawing_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # Clear drawing mask
        draw_points = []  # Clear stored points
        drawing_paths = []  # Clear all paths
        current_path = []  # Clear current path
        print("ALL drawings cleared including permanent ones")
    elif key == ord('p'):  # Toggle persistent drawing visibility
        show_persistent_drawing = not show_persistent_drawing
        print(f"Persistent drawing visibility: {'ON' if show_persistent_drawing else 'OFF'}")
    elif key == ord('m'):  # Switch modes
        if drawing_mode == "overlay":
            drawing_mode = "whiteboard"
        else:
            drawing_mode = "overlay"
        print(f"Switched to {drawing_mode} mode")
    elif key == ord('s'):  # Save the drawing
        filename = f"drawing_{int(time.time())}.png"
        if drawing_mode == "overlay":
            # Save the persistent canvas instead of just the current view
            cv2.imwrite(filename, persistent_canvas)
        else:
            # Save the combined image with persistent drawings
            cv2.imwrite(filename, canvas_draw)
        print(f"Drawing saved as {filename}")
    elif key == ord('+') or key == ord('='):  # Increase brush size
        if brush_thickness < max_brush_thickness:
            brush_thickness += brush_size_increment
            print(f"Brush size increased to {brush_thickness}")
    elif key == ord('-') or key == ord('_'):  # Decrease brush size
        if brush_thickness > min_brush_thickness:
            brush_thickness -= brush_size_increment
            print(f"Brush size decreased to {brush_thickness}")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 