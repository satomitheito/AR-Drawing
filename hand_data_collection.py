import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# Create directory for saving data if it doesn't exist
os.makedirs('model/keypoint_classifier', exist_ok=True)
csv_path = 'model/keypoint_classifier/keypoint.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define gesture labels
GESTURE_LABELS = {
    0: "size up",
    1: "size down",
    2: "nothing",
    3: "erase",
    4: "point",
    5: "color",
    6: "random",
}

# Create a labels file if it doesn't exist
label_file_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
if not os.path.exists(label_file_path):
    with open(label_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in GESTURE_LABELS.items():
            writer.writerow([value])
    print(f"Created labels file at {label_file_path}")

# Variables for data collection
mode = 0  # 0: Detection, 1: Data collection
gesture_id = -1  # Gesture class number
capture_ready = False  # Flag to indicate if ready to capture data

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

# Function to save captured data
def save_landmark_data(gesture_id, processed_landmark_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id] + processed_landmark_list)
    print(f"Captured gesture {GESTURE_LABELS[gesture_id]}")

while cap.isOpened():
    # Read frame from webcam
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break
    
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    debug_image = image.copy()
    
    # Convert to RGB and process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Initialize landmark_list
    landmark_list = []
    processed_landmark_list = []
    handedness_label = ""
    
    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                             results.multi_handedness if results.multi_handedness else []):
            # Draw landmarks
            mp_drawing.draw_landmarks(
                debug_image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Extract landmark coordinates
            image_height, image_width, _ = image.shape
            landmark_list = []
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_list.append([landmark_x, landmark_y])
            
            # Pre-process landmarks
            processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Display handedness
            handedness_label = handedness.classification[0].label  # 'Left' or 'Right'
            cv2.putText(debug_image, f"Hand: {handedness_label}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # In collection mode, any hand is valid for gesture capture
            if mode == 1 and 0 <= gesture_id < len(GESTURE_LABELS):
                cv2.putText(debug_image, f"Ready to capture: {GESTURE_LABELS[gesture_id]}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                capture_ready = True
    else:
        capture_ready = False
    
    # Display mode and instructions
    if mode == 1:
        cv2.putText(debug_image, "MODE: Data collection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if 0 <= gesture_id < len(GESTURE_LABELS):
            cv2.putText(debug_image, f"GESTURE: {GESTURE_LABELS[gesture_id]}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(debug_image, "MODE: Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display keypress instructions
    cv2.putText(debug_image, "Press 'c' to toggle collection mode", (10, image.shape[0] - 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_image, "Use number keys to select gesture:", (10, image.shape[0] - 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_image, "0-6: gestures (0: size up, 1: size down, ..., 6: random)", (10, image.shape[0] - 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_image, "Press SPACE to capture the current hand gesture", (10, image.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the image
    cv2.imshow('Hand Gesture Data Collection', debug_image)
    
    # Key handling
    key = cv2.waitKey(10)
    if key & 0xFF == 27:  # ESC
        break
    elif key == ord('c'):  # Toggle collection mode
        mode = 0 if mode == 1 else 1
    elif 48 <= key <= 54:  # 0-6 for the seven gestures
        gesture_id = key - 48
    elif key == 32 and mode == 1 and capture_ready:  # SPACE key for manual capture
        if processed_landmark_list:
            save_landmark_data(gesture_id, processed_landmark_list)
            # Visual feedback for capture
            cv2.putText(debug_image, "CAPTURED!", (image.shape[1]//2-100, image.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Hand Gesture Data Collection', debug_image)
            cv2.waitKey(500)  # Show the "CAPTURED!" text for 500ms

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"Data collection complete. Data saved to {csv_path}")
print(f"Labels saved to {label_file_path}")