import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

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

def draw_hand_landmarks(image, landmarks):
    """Draw hand landmarks with connections"""
    # Draw dots at each landmark
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    # Define connections between landmarks (simplified)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw lines for connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
            end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

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

# Define index finger tip index
INDEX_FINGER_TIP = 8  # MediaPipe hand landmark index for index finger tip

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
            
            # If "point" gesture is detected, highlight the index finger tip
            if predicted_label == "point" and confidence > 0.7:  # You can adjust this threshold
                # Get index finger tip coordinates
                index_finger_tip = landmark_list[INDEX_FINGER_TIP]
                x, y = index_finger_tip
                
                # Draw a more visible highlight at the index finger tip
                # Inner solid circle (red)
                cv2.circle(image, (x, y), 12, (0, 0, 255), -1)
                # Middle ring (white)
                cv2.circle(image, (x, y), 18, (255, 255, 255), 3)
                # Outer ring (blue)
                cv2.circle(image, (x, y), 24, (255, 0, 0), 2)
                
                # Draw a ray/line from the tip to show direction
                ray_length = 40
                cv2.line(image, (x, y), (x, y - ray_length), (0, 255, 255), 4)
                
                # Display fingertip coordinates with more visibility
                tip_text = f"Index Tip: ({x}, {y})"
                cv2.putText(image, tip_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
    
    # Display the hand landmarks
    cv2.putText(image, "Press ESC to exit", (10, image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show the image
    cv2.imshow('Hand Gesture Recognition', image)
    
    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 