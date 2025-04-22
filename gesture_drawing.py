import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# Load the TFLite model
model_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

# Load gesture labels
label_file_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
if os.path.exists(label_file_path):
    with open(label_file_path, 'r') as f:
        labels = [line.strip() for line in f]
else:
    # Default labels if file not found
    labels = ["size up", "size down", "nothing", "erase", "point", "color", "random"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas for drawing
ret, frame = cap.read()
if not ret:
    print("Failed to grab initial frame")
    exit()

# Create a white canvas instead of transparent
canvas = np.ones_like(frame) * 255  # White background
draw_overlay = np.zeros_like(frame)  # Transparent overlay for combined view

# Drawing settings
draw_color = (0, 0, 255)  # Red - more visible
brush_thickness = 20  # Increased thickness even more for visibility
is_drawing = False
last_point = None
draw_points = []  # Store all points for debugging
smoothing_points = []  # For line smoothing
smoothing_window = 2  # Reduced for more immediate response

# Initialize drawing paths - store separate strokes
drawing_paths = []
current_path = []

# Drawing mode
drawing_mode = "overlay"  # Can be "overlay" or "whiteboard"

# Define index finger tip index
INDEX_FINGER_TIP = 8  # MediaPipe hand landmark index for index finger tip

def pre_process_landmark(landmark_list):
    """Convert landmark coordinates to relative coordinates and normalize"""
    temp_landmark_list = landmark_list.copy()
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for i, point in enumerate(temp_landmark_list):
        if i == 0:  # Use the wrist as the base point
            base_x, base_y = point[0], point[1]
        
        temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x
        temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y
    
    # Flatten the list
    temp_landmark_list = list(np.array(temp_landmark_list).flatten())
    
    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value != 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    
    return temp_landmark_list

def predict_gesture(landmark_list):
    """Predict gesture using the TFLite model"""
    # Reshape input data to match model's expected shape
    input_data = np.array([landmark_list], dtype=np.float32)
    input_data = input_data.reshape(1, 21, 2)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the index of the highest probability
    predicted_class_idx = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class_idx]
    
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

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam")
        break
    
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
                h, w, c = image.shape
                x = min(int(landmark.x * w), w - 1)
                y = min(int(landmark.y * h), h - 1)
                landmark_list.append([x, y])
            
            # Process landmarks for classification
            processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Reshape landmarks for our model
            landmark_array = np.array(processed_landmark_list).reshape(21, 2)
            
            # Predict gesture
            predicted_class_idx, confidence = predict_gesture(landmark_array)
            if predicted_class_idx < len(labels):
                predicted_label = labels[predicted_class_idx]
            else:
                predicted_label = f"Unknown ({predicted_class_idx})"
            
            # Display the gesture label with confidence
            info_text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(image, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If "point" gesture is detected, highlight the index finger tip and draw
            if predicted_label == "point" and confidence > 0.7:  # You can adjust this threshold
                # Get index finger tip coordinates
                index_finger_tip = landmark_list[INDEX_FINGER_TIP]
                x, y = index_finger_tip
                
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
                tip_text = f"Drawing at: ({x}, {y})"
                cv2.putText(image, tip_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                # If not pointing, reset drawing state
                is_drawing = False
                last_point = None
    
    # Handle drawing logic
    if pointing_this_frame:
        if not is_drawing:
            # Start a new drawing stroke
            is_drawing = True
            last_point = current_point
            smoothing_points = [current_point]  # Reset smoothing
            current_path = []  # Start a new path
            # Debug message
            print(f"Starting to draw at {current_point}")
        elif last_point is not None and last_point != current_point:  # Avoid drawing the same point
            # Get smoothed point for more natural lines
            smoothed_point = get_smoothed_point(current_point)
            
            # Continue the stroke by drawing a line from last point to current
            # Draw directly on the image for immediate feedback
            cv2.line(image, last_point, smoothed_point, (255, 255, 255), brush_thickness + 6, cv2.LINE_AA)  # White outline
            cv2.line(image, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)  # Color line
            
            # Add to the persistent canvases
            cv2.line(canvas, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)
            cv2.line(draw_overlay, last_point, smoothed_point, draw_color, brush_thickness, cv2.LINE_AA)
            
            # Store the point in both lists
            draw_points.append(smoothed_point)
            current_path.append(smoothed_point)
            
            # Update last point
            last_point = smoothed_point
            
            # Debug info
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
        # Not pointing, finish the current stroke if we were drawing
        if is_drawing and current_path:
            drawing_paths.append(current_path)  # Store the completed path
            current_path = []
        
        # Reset drawing state
        is_drawing = False
        last_point = None
        smoothing_points = []  # Reset smoothing
    
    # Update both canvases every frame with all paths
    visible_overlay = np.zeros_like(frame)

    # Draw all completed paths
    for path in drawing_paths:
        if len(path) > 1:
            for i in range(1, len(path)):
                # Draw thick white outline first
                cv2.line(visible_overlay, path[i-1], path[i], 
                        (255, 255, 255), brush_thickness + 6, cv2.LINE_AA)
                # Then draw colored line on top
                cv2.line(visible_overlay, path[i-1], path[i], 
                        draw_color, brush_thickness, cv2.LINE_AA)

    # Draw current path if active
    if current_path and len(current_path) > 1:
        for i in range(1, len(current_path)):
            # Draw thick white outline first
            cv2.line(visible_overlay, current_path[i-1], current_path[i], 
                    (255, 255, 255), brush_thickness + 6, cv2.LINE_AA)
            # Then draw colored line on top
            cv2.line(visible_overlay, current_path[i-1], current_path[i], 
                    draw_color, brush_thickness, cv2.LINE_AA)

    # Choose which display mode to use
    if drawing_mode == "overlay":
        # Combine the drawing overlay with the camera image (already has lines drawn directly)
        combined_image = cv2.addWeighted(image, 1.0, visible_overlay, 1.0, 0)
    else:  # whiteboard mode
        # Redraw all paths on a fresh canvas each frame
        canvas_draw = np.ones_like(frame) * 255  # Fresh white canvas
        
        # Draw all completed paths
        for path in drawing_paths:
            if len(path) > 1:
                for i in range(1, len(path)):
                    cv2.line(canvas_draw, path[i-1], path[i], 
                            draw_color, brush_thickness, cv2.LINE_AA)
        
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
    if pointing_this_frame:
        status_text = "DRAWING ACTIVE"
        cv2.putText(combined_image, status_text, (combined_image.shape[1] - 250, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # Add a visual indicator that drawing is happening
        cv2.circle(combined_image, (combined_image.shape[1] - 30, 30), 15, (0, 255, 0), -1)

    # Display UI controls
    cv2.putText(combined_image, "Press 'c' to clear drawing", (10, combined_image.shape[0] - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 'm' to switch modes", (10, combined_image.shape[0] - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press 's' to save drawing", (10, combined_image.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined_image, "Press ESC to exit", (10, combined_image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Show the combined image
    cv2.imshow('AR Drawing with Hand Gestures', combined_image)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  # Clear the canvas
        canvas = np.ones_like(frame) * 255  # White background
        draw_overlay = np.zeros_like(frame)  # Clear overlay
        draw_points = []  # Clear stored points
        drawing_paths = []  # Clear all paths
        current_path = []  # Clear current path
        print("Canvas cleared")
    elif key == ord('m'):  # Switch modes
        if drawing_mode == "overlay":
            drawing_mode = "whiteboard"
        else:
            drawing_mode = "overlay"
        print(f"Switched to {drawing_mode} mode")
    elif key == ord('s'):  # Save the drawing
        filename = f"drawing_{int(time.time())}.png"
        if drawing_mode == "overlay":
            cv2.imwrite(filename, draw_overlay)
        else:
            cv2.imwrite(filename, canvas)
        print(f"Drawing saved as {filename}")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 