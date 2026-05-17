from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Road Condition Detection", annotated_frame)

    # Press ESC to close
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()