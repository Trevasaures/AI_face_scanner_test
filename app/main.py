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

'''
Training models were created outside of Insco by the following authors under the Intel License Agreement:
Rainer Lienhart

The model has not been trained beyond what the original authors have done.

We have only used the model for demonstration purposes in this script.
'''

# Importing the necessary libraries 
import cv2
import tensorflow as tf
import numpy as np
import random
import time

from emotion_colors import emotion_colors
from screen_utils import get_scaled_dimensions

# Constants
EMOTION_MODEL_PATH = '../data/emotion_model.hdf5'
HAARCASCADE_PATH = '../data/haarcascade_frontalface_default.xml'
LOGO_PATH = '../data/insco-white-halo.png'
MAX_CAMERAS = 5
LOGO_SIZE = (100, 100)
FRAME_SKIP = 3
EMOTION_DISPLAY_DURATION = 30

# Load resources
def load_resources():
    emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH, compile=False)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    logo = cv2.resize(logo, LOGO_SIZE)
    return emotion_model, face_cascade, process_logo(logo)

# Process the logo for overlay
def process_logo(logo):
    if logo.shape[2] == 4:  # Transparency handling
        alpha_channel = logo[:, :, 3] / 255.0
        logo_rgb = logo[:, :, :3]
    else:
        alpha_channel = np.ones((logo.shape[0], logo.shape[1]))
        logo_rgb = logo
    return logo_rgb, alpha_channel

# Detect available cameras
def get_available_cameras(max_index=MAX_CAMERAS):
    return [index for index in range(max_index) if cv2.VideoCapture(index).isOpened()]

# Prompt user for camera selection
def select_camera(available_cameras):
    print("\nAvailable cameras:")
    for i, cam_index in enumerate(available_cameras):
        description = "Embedded webcam" if cam_index == 0 else f"External camera {i}"
        print(f"{i}: Camera index {cam_index} ({description})")
    while True:
        try:
            selected_index = int(input("\nSelect the camera index to use: "))
            if selected_index in range(len(available_cameras)):
                return available_cameras[selected_index]
            print(f"Invalid selection. Please choose a number between 0 and {len(available_cameras) - 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main frame processing loop
def process_frames(cap, emotion_model, face_cascade, logo_rgb, alpha_channel):
    emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    fun_emotions = ["Stinky", "Sleepy", "Excited", "Confused", "Cheesy", "Goofy", "Zoned Out"]
    frame_index = 0
    current_emotion_label = "Neutral"
    current_color = emotion_colors.get(current_emotion_label, (255, 255, 255))
    emotion_hold_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_index += 1
        if frame_index % FRAME_SKIP != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            if emotion_hold_counter == 0:
                current_emotion_label, current_color = detect_emotion(emotion_model, face_roi, emotion_classes, fun_emotions)
                emotion_hold_counter = EMOTION_DISPLAY_DURATION

            emotion_hold_counter -= 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, 2)
            cv2.putText(frame, f"{current_emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

        overlay_logo(frame, logo_rgb, alpha_channel)
        cv2.imshow("Employee Emotions (the camera don't lie)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Session terminated smoothly.")
            break

# Detect emotion from face ROI
def detect_emotion(emotion_model, face_roi, emotion_classes, fun_emotions):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) / 255.0
    face_expanded = np.expand_dims(np.expand_dims(face_gray, axis=-1), axis=0)
    emotion_prediction = emotion_model.predict(face_expanded, verbose=0)
    confidence = np.max(emotion_prediction)
    if confidence > 0.5:
        label = emotion_classes[np.argmax(emotion_prediction)]
    else:
        label = random.choice(fun_emotions)
    color = emotion_colors.get(label, (255, 255, 255))
    return label, color

# Overlay logo on frame
def overlay_logo(frame, logo_rgb, alpha_channel):
    h, w, _ = logo_rgb.shape
    overlay = frame[10:10 + h, 10:10 + w]
    for c in range(3):
        frame[10:10 + h, 10:10 + w, c] = (alpha_channel * logo_rgb[:, :, c] +
                                          (1 - alpha_channel) * overlay[:, :, c])

# Main program entry
if __name__ == "__main__":
    emotion_model, face_cascade, (logo_rgb, alpha_channel) = load_resources()
    available_cameras = get_available_cameras()

    if not available_cameras:
        print("No cameras detected. Please connect a camera and try again.")
        exit()

    selected_camera = select_camera(available_cameras)
    cap = cv2.VideoCapture(selected_camera)
    print(f"\nUsing camera index {selected_camera}")

    video_width, video_height = get_scaled_dimensions(scale=0.8)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)

    print(f"Video resolution: {video_width}x{video_height}")
    time.sleep(2)

    process_frames(cap, emotion_model, face_cascade, logo_rgb, alpha_channel)
    cap.release()
    cv2.destroyAllWindows()

