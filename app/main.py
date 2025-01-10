'''
This is a simple script that uses a webcam to detect faces and predict emotions in real-time.

The script uses a pre-trained Convolutional Neural Network (CNN) model to predict emotions from facial expressions.

The emotions detected are: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

The script also injects some fun emotions randomly to make it more interesting.

The script uses OpenCV for webcam access and face detection.

The script uses TensorFlow and Keras for emotion prediction.

The script uses a Haarcascade classifier for face detection.

There are some fun emotions injected randomly to make the script more interesting.

Enjoy the script and have fun with the emotions!
'''

# Importing the necessary libraries 
import cv2
import tensorflow as tf
import numpy as np
import random
import time

from emotion_colors import emotion_colors
from screen_utils import get_scaled_dimensions

# Load the emotion detection model
emotion_model_path = '../data/emotion_model.hdf5'
emotion_model = tf.keras.models.load_model(emotion_model_path, compile=False)
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# List of fun emotions
fun_emotions = ["Stinky", "Sleepy", "Excited", "Confused", "Cheesy", "Goofy", "Zoned Out"]

haarcascade_path = '../data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Load logo
logo_path = '../data/insco-white-halo.png'  
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

# Resize the logo
logo = cv2.resize(logo, (100, 100)) # width / height

# Separate alpha channel (if the logo has transparency)
if logo.shape[2] == 4:
    alpha_channel = logo[:, :, 3] / 255.0
    logo_rgb = logo[:, :, :3]
else:
    alpha_channel = np.ones((logo.shape[0], logo.shape[1]))
    logo_rgb = logo

# Try multiple webcam inputs
for cam_index in range(5):
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print("\nIf index is 0, the program is using your device's embedded webcam.")
        print(f"Using webcam index {cam_index}\n")
        break
else:
    print("Error: Could not open any webcam. Exiting...")
    exit()

# Dynamically set the resolution to 80% of the screen size
video_width, video_height = get_scaled_dimensions(scale=0.8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

print(f"Video resolution: {video_width}x{video_height}")

# Wait for the webcam to initialize
time.sleep(2)

# Frame processing control
frame_index = 0
frame_skip = 3
emotion_display_duration = 30
current_emotion_label = "Neutral"
current_color = emotion_colors.get(current_emotion_label, (255, 255, 255))
emotion_hold_counter = 0

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

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        if emotion_hold_counter == 0:
            face_resized = cv2.resize(face_roi, (64, 64))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_normalized = face_gray / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)
            face_expanded = np.expand_dims(face_expanded, axis=-1)

            emotion_prediction = emotion_model.predict(face_expanded, verbose=0)
            confidence = np.max(emotion_prediction)
            if confidence > 0.5:
                current_emotion_label = emotion_classes[np.argmax(emotion_prediction)]
            else:
                current_emotion_label = "Neutral"

            if frame_index % 5 == 0 or frame_index % 7 == 0:
                current_emotion_label = random.choice(fun_emotions)

            current_color = emotion_colors.get(current_emotion_label, (255, 255, 255))
            emotion_hold_counter = emotion_display_duration

        emotion_hold_counter -= 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, 2)
        cv2.putText(frame, f"{current_emotion_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

    # Overlay the logo in the top-left corner
    h, w, _ = logo_rgb.shape
    overlay = frame[10:10 + h, 10:10 + w]  # Adjust position as needed
    for c in range(3):  # Apply overlay channel-wise
        frame[10:10 + h, 10:10 + w, c] = (alpha_channel * logo_rgb[:, :, c] +
                                          (1 - alpha_channel) * overlay[:, :, c])

    # Display the frame
    cv2.imshow("Employee Emotions (the camera don't lie)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session terminated smoothly.")
        break

cap.release()
cv2.destroyAllWindows()
