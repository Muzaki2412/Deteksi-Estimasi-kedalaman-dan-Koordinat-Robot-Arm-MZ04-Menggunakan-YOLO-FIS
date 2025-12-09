import cv2
from ultralytics import YOLO
import math

# Load the YOLOv8n model
model = YOLO("bestsegm (2).pt")

# Open the webcam
cap = cv2.VideoCapture(1) #5, cv2.CAP_DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit the tracking")

# Loop through the webcam frames
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        
        # Count detected objects
        count_objects = 0
        if results[0].boxes is not None:
            count_objects = len(results[0].boxes)
        
        # Display count on the frame
        cv2.putText(annotated_frame, f"Objects: {count_objects}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8n Webcam Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Failed to grab frame from webcam")
        break

cap.release()
cv2.destroyAllWindows()
print("Tracking stopped")
