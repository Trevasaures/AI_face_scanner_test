# Webcam and AI test script
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque

# Load the emotion detection model
emotion_model_path = './data/emotion_model.hdf5'
emotion_model = tf.keras.models.load_model(emotion_model_path, compile=False)  # Load model without compiling
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load face detection model
haarcascade_path = './data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Emotion history buffer
emotion_history = deque(maxlen=10) 

# Frame processing control
frame_index = 0
frame_skip = 3  # Process every 3rd frame for better responsiveness

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_index += 1
    if frame_index % frame_skip != 0:  # Skip frames for reduced sensitivity
        continue

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using Haarcascade
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]  # Extract face ROI
        if face_roi.size == 0:
            continue

        # Preprocess face for emotion model
        face_resized = cv2.resize(face_roi, (64, 64))  # Resize to 64x64
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_normalized = face_gray / 255.0  # Normalize to [0, 1]
        face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
        face_expanded = np.expand_dims(face_expanded, axis=-1)  # Add channel dimension

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_expanded, verbose=0)  # Suppress model logs
        confidence = np.max(emotion_prediction)  # Get confidence of the prediction
        if confidence > 0.5:  # Use only confident predictions
            emotion_label = emotion_classes[np.argmax(emotion_prediction)]
            emotion_history.append(emotion_label)  # Add to history

        # Get the most common emotion in history
        if len(emotion_history) > 0:
            stable_emotion = max(set(emotion_history), key=emotion_history.count)
        else:
            stable_emotion = "Neutral"  # Default if no stable emotion detected

        # Draw bounding box and stable emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, stable_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display emotion history on the frame
    history_text = "History: " + ", ".join(emotion_history)
    cv2.putText(frame, history_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session terminated smoothly.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


