# Webcam and AI test script
import cv2
import tensorflow as tf
import numpy as np
from emotion_colors import emotion_colors
import time

# Load the emotion detection model
emotion_model_path = '../data/emotion_model.hdf5'
emotion_model = tf.keras.models.load_model(emotion_model_path, compile=False)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load face detection model
haarcascade_path = '../data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Try multiple webcam indices
for cam_index in range(5):  # Test indices 0 to 4
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Using webcam index {cam_index}")
        break
else:
    print("Error: Could not open any webcam. Exiting...")
    exit()

# Set camera resolution (fallback to defaults if unsupported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Wait for the webcam to initialize
time.sleep(2)

# Frame processing control
frame_index = 0
frame_skip = 1  # Process every frame for higher accuracy

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_index += 1
    if frame_index % frame_skip != 0:
        continue

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haarcascade
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        # Preprocess face for emotion model
        face_resized = cv2.resize(face_roi, (64, 64))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
        face_expanded = np.expand_dims(face_expanded, axis=-1)   # Add channel dimension

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_expanded, verbose=0)
        confidence = np.max(emotion_prediction)
        if confidence > 0.5:
            emotion_label = emotion_classes[np.argmax(emotion_prediction)]
        else:
            emotion_label = "Neutral"

        # Get color for the emotion
        color = emotion_colors.get(emotion_label, (255, 255, 255))

        # Draw bounding box and emotion label with color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{emotion_label} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session terminated smoothly.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
