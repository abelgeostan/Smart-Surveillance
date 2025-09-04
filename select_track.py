import cv2
import time
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720
OUTPUT_FILENAME = "output_video.mp4"
# ---------------------

# --- State Management Variables ---
target_id = None
trail_points = []
# A global list to store current boxes and IDs for the mouse callback
current_boxes_with_ids = []
# --------------------------------

# --- Mouse Callback Function ---
# This function will be called every time there is a mouse event
def select_person(event, x, y, flags, param):
    global target_id, trail_points
    # Check if the event is a left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Loop through the boxes detected in the current frame
        for box, track_id in current_boxes_with_ids:
            x1, y1, x2, y2 = box
            # Check if the click coordinates are inside a bounding box
            if x1 < x < x2 and y1 < y < y2:
                target_id = track_id
                trail_points = [] # Reset trail for the new target
                print(f"Target person selected. ID: {target_id}")
                break # Exit the loop once a target is found
# -----------------------------

# Load the YOLOv11-Pose model
model = YOLO('yolo11m-pose.pt')

# Path to your video file
video_path = "gud_video.mp4"
cap = cv2.VideoCapture(video_path)

# Create a resizable window and set the mouse callback
window_name = "YOLOv11 Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, select_person)

# --- Video Writer Setup ---
# Get video properties for the writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
fps_out = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (RESIZE_WIDTH, RESIZE_HEIGHT)
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps_out, frame_size)
# --------------------------

# Create a black image (mask) for the trail
trail_mask = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)

# FPS Counter variables
prev_time = 0

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize the Frame
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Run YOLOv8 tracking
    results = model.track(frame, tracker="botsort.yaml", persist=True)

    # Prepare data for mouse callback and drawing
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        current_boxes_with_ids = list(zip(boxes, track_ids))
    else:
        current_boxes_with_ids = []

    # --- Drawing and Trail Logic ---
    # If no one is selected yet, draw all boxes
    if target_id is None:
        for box, track_id in current_boxes_with_ids:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instruction text
        cv2.putText(frame, "Click on a person to select and track", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If a target is selected, only draw the target and their trail
    else:
        target_found = False
        for box, track_id in current_boxes_with_ids:
            if track_id == target_id:
                target_found = True
                x1, y1, x2, y2 = box
                
                # Draw the bounding box for the target
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Highlight in yellow
                label = f"Tracking ID: {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Calculate the leg point (bottom center of the box)
                leg_point = ((x1 + x2) // 2, y2)
                
                # Add the new point to our trail list
                trail_points.append(leg_point)
                
                # Draw the trail on the mask
                for i in range(1, len(trail_points)):
                    cv2.line(trail_mask, trail_points[i-1], trail_points[i], (255, 0, 0), 2) # Blue trail
                break # Exit loop once target is found and processed
        
        if not target_found:
             cv2.putText(frame, f"ID {target_id} Lost", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Combine the frame with the trail mask
    # This overlays the persistent trail onto the current frame
    final_frame = cv2.add(frame, trail_mask)
    
    # Display FPS
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(final_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow(window_name, final_frame)
    
    # Write the frame to the output video file
    out.write(final_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
print(f"Processing finished. Output video saved as {OUTPUT_FILENAME}")
cap.release()
out.release()
cv2.destroyAllWindows()