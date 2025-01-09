import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if ret:
    cv2.imshow("Test Frame", frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
cap.release()
