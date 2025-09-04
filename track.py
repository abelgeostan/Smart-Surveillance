import cv2
import time
from ultralytics import YOLO

# --- Configuration ---
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720
# ---------------------

# Load the YOLOv8-Pose model
model = YOLO('yolo11x-pose.pt')

# Path to your video file
video_path = "gud_video.mp4"
cap = cv2.VideoCapture(video_path)

# Create a resizable window
window_name = "YOLOv8 Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# FPS Counter variables
prev_time = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the Frame
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, tracker="botsort.yaml", persist=True)

        # --- Manual Drawing Section ---
        # Get the bounding boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Loop through each detected person
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare the label text with the track ID
            label = f"ID: {track_id}"
            
            # Put the label text above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # ----------------------------

        # Display FPS on the frame
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with manual annotations
        # Note: We are now displaying 'frame' directly, not 'annotated_frame'
        cv2.imshow(window_name, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()