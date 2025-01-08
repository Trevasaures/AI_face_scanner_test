# Webcam and AI test script
import tensorflow as tf
import cv2

# Verify TensorFlow and OpenCV installations
print(f"TensorFlow version: {tf.__version__}")
print("OpenCV is working!")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Display the webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()