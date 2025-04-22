import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Define paths
dataset_path = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier_1d.keras'
label_file_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'

# Load gesture labels
if os.path.exists(label_file_path):
    with open(label_file_path, 'r') as f:
        labels = [line.strip() for line in f]
    print(f"Loaded {len(labels)} label names from {label_file_path}")
else:
    # Default labels if file not found
    labels = ["size up", "size down", "nothing", "erase", "point", "color", "random"]
    print("Using default labels")

# Number of classes
NUM_CLASSES = len(labels)

print(f"Training a 1D CNN model for {NUM_CLASSES} gesture classes: {labels}")

# Load dataset
try:
    data = np.loadtxt(dataset_path, delimiter=',')
    print(f"Loaded {len(data)} samples from {dataset_path}")
    
    # Split into features and labels
    X_dataset = data[:, 1:]  # All columns except first (keypoints)
    y_dataset = data[:, 0].astype(int)  # First column (labels)
    
    # Check class distribution
    unique_classes, counts = np.unique(y_dataset, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique_classes, counts):
        label_name = labels[cls] if cls < len(labels) else f"Class {cls}"
        print(f"  {label_name} (ID {cls}): {count} samples")
    
    # The original dataset is in the shape expected for 2D CNN
    # For 1D CNN, we just flatten the data differently
    # Each landmark has x,y coords, so we have 21 landmarks * 2 = 42 features per sample
    num_landmarks = 21
    
    # For 1D CNN, we keep data in this flattened format
    # This is different from the 2D CNN reshape which was (samples, 21, 2)
    # No need to reshape here, as the data is already in flattened format
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, 
        y_dataset, 
        train_size=0.8, 
        random_state=42,
        stratify=y_dataset  # Maintain class distribution in train/test split
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Define 1D CNN model
    def create_1d_cnn_model():
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Create model
    model = create_1d_cnn_model()
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, 
            monitor='val_accuracy',
            verbose=1, 
            save_best_only=True,
            save_weights_only=False
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20, 
            verbose=1,
            restore_best_weights=True
        ),
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Generate and visualize predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    label_names = [labels[i] if i < len(labels) else f"Class {i}" for i in range(NUM_CLASSES)]
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    report = classification_report(
        y_test, 
        y_pred_classes, 
        target_names=label_names,
        digits=4
    )
    print("Classification Report:")
    print(report)
    
    # Feature importance analysis (optional)
    # This is a simple way to visualize which features are more important
    # Get weights from the first layer
    if hasattr(model.layers[1], 'get_weights') and len(model.layers[1].get_weights()) > 0:
        # Get weights from the first Conv1D layer
        first_layer_weights = model.layers[1].get_weights()[0]
        
        # Average across filters and the kernel size
        feature_importance = np.mean(np.abs(first_layer_weights), axis=(1, 2))
        
        # Print shape info for debugging
        print(f"Feature importance shape: {feature_importance.shape}")
        
        # Check if we can reshape to the expected landmark format
        if len(feature_importance) == 42:  # 21 landmarks * 2 coordinates
            # Reshape to landmarks (21 landmarks, each with x,y)
            feature_importance_reshaped = feature_importance.reshape(21, 2)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(feature_importance_reshaped, cmap='viridis', aspect='auto')
            plt.colorbar(label='Average Weight Magnitude')
            plt.xlabel('Coordinate (0=x, 1=y)')
            plt.ylabel('Landmark Index')
            plt.title('Landmark Feature Importance')
            plt.tight_layout()
            plt.show()
            
            # Plot overall importance of each landmark (combining x,y)
            landmark_importance = np.mean(feature_importance_reshaped, axis=1)
            plt.figure(figsize=(12, 6))
            plt.bar(range(21), landmark_importance)
            plt.xlabel('Landmark Index')
            plt.ylabel('Importance Score')
            plt.title('Relative Importance of Each Hand Landmark')
            plt.xticks(range(21))
            plt.tight_layout()
            plt.show()
            
            # Print the most important landmarks
            top_landmarks = np.argsort(landmark_importance)[::-1][:5]
            print("\nTop 5 most important landmarks:")
            for i, idx in enumerate(top_landmarks):
                print(f"{i+1}. Landmark {idx}: {landmark_importance[idx]:.4f}")
        else:
            # If we can't reshape to landmarks, just visualize as a flat array
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(feature_importance)), feature_importance)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
            # Print the most important features
            top_features = np.argsort(feature_importance)[::-1][:5]
            print("\nTop 5 most important features:")
            for i, idx in enumerate(top_features):
                print(f"{i+1}. Feature {idx}: {feature_importance[idx]:.4f}")

    print(f"\nModel successfully trained and saved to {model_save_path}")
    print("You can now use this model with the gesture_recognition_1d.py script")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 