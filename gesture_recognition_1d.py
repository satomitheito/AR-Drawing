import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

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

NUM_CLASSES = len(labels)

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
    
    # Flatten the list for 1D CNN (x1, y1, x2, y2, ..., x21, y21)
    temp_landmark_list = list(np.array(temp_landmark_list).flatten())
    
    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value != 0:
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    
    return temp_landmark_list

def create_1d_cnn_model():
    """Create a 1D CNN model for hand gesture recognition"""
    # Input shape will be (42,) - flattened x,y coordinates of 21 landmarks
    model = tf.keras.Sequential([
        # Reshape input to (42, 1) for Conv1D
        tf.keras.layers.Reshape((42, 1), input_shape=(42,)),
        
        # First Conv1D block
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Second Conv1D block
        tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Third Conv1D block
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        # Global pooling and classification
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create or load model
model_path = 'model/keypoint_classifier/keypoint_classifier_1d.keras'
if os.path.exists(model_path):
    # Load existing model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded existing model from {model_path}")
else:
    # Create new model
    model = create_1d_cnn_model()
    print("Created new 1D CNN model")
    
    # Since this is a new model, we need to save it for future use
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved new model to {model_path}")

def predict_gesture(landmark_list):
    """Predict gesture using the 1D CNN model"""
    # For 1D CNN, input is a flattened array of 42 values (21 landmarks x 2 coordinates)
    input_data = np.array([landmark_list], dtype=np.float32)
    
    # Make prediction
    predictions = model.predict(input_data, verbose=0)
    
    # Get the index of the highest probability
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class_idx, confidence

def train_model_with_sample(landmark_list, true_class_idx):
    """Update model with a single training sample"""
    # Prepare input data
    X = np.array([landmark_list], dtype=np.float32)
    y = np.array([true_class_idx], dtype=np.int32)
    
    # Train for a single step
    model.fit(X, y, epochs=1, verbose=0)
    
    # Save the updated model
    model.save(model_path)
    print(f"Model updated with sample for class: {labels[true_class_idx]}")

# Main loop
training_mode = False
current_class = 0

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
    
    # Display mode status
    mode_text = "TRAINING MODE" if training_mode else "RECOGNITION MODE"
    cv2.putText(image, mode_text, (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    if training_mode:
        cv2.putText(image, f"Current class: {labels[current_class]}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
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
            
            # Process landmarks for classification (1D format)
            processed_landmark_list = pre_process_landmark(landmark_list)
            
            if training_mode:
                # In training mode, display capture instructions
                cv2.putText(image, "Press SPACE to capture sample", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # In recognition mode, make prediction
                predicted_class_idx, confidence = predict_gesture(processed_landmark_list)
                if predicted_class_idx < len(labels):
                    predicted_label = labels[predicted_class_idx]
                else:
                    predicted_label = f"Unknown ({predicted_class_idx})"
                
                # Display the gesture label with confidence
                info_text = f"{predicted_label} ({confidence:.2f})"
                cv2.putText(image, info_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display instructions
    cv2.putText(image, "T: Toggle training mode", (10, image.shape[0] - 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "0-6: Select class in training mode", (10, image.shape[0] - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "SPACE: Capture sample in training mode", (10, image.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "ESC: Exit", (10, image.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show the image
    cv2.imshow('1D CNN Hand Gesture Recognition', image)
    
    # Key handling
    key = cv2.waitKey(5) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('t'):  # Toggle training mode
        training_mode = not training_mode
        print(f"{'Training' if training_mode else 'Recognition'} mode activated")
    elif 48 <= key <= 54:  # 0-6 keys for class selection in training mode
        if training_mode:
            current_class = key - 48
            if current_class < len(labels):
                print(f"Selected class: {labels[current_class]}")
            else:
                print(f"Selected class index {current_class} is out of range")
                current_class = 0
    elif key == 32:  # SPACE to capture sample in training mode
        if training_mode and results.multi_hand_landmarks:
            # Train model with current sample
            train_model_with_sample(processed_landmark_list, current_class)

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 