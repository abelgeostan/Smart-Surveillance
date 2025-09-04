import cv2
import time
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# --- Configuration ---
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720
VIDEO_PATH = "test_video.mp4"
OUTPUT_FILENAME = "trail_output.avi"
# ---------------------

# Load the YOLOv8-Pose model
model = YOLO('yolo11x-pose.pt')

# =================================================================================
# PASS 1: VISUAL ANALYSIS TO IDENTIFY TRACKS
# =================================================================================
print("Starting Pass 1: Analyzing video. Watch the video and note the ID you want to track.")
print("Press 'q' to skip to the end of the analysis.")

cap = cv2.VideoCapture(VIDEO_PATH)

# This dictionary will store all data
all_frames_data = defaultdict(list)
frame_index = 0

# Create a window for the visual analysis
cv2.namedWindow("Pass 1: Analysis", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    
    # Run tracker
    results = model.track(frame, tracker="botsort.yaml", persist=True, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # Save data and draw all detections for user to see
        for box, track_id in zip(boxes, track_ids):
            all_frames_data[frame_index].append({"id": track_id, "box": box})
            
            # Draw the box and ID on the frame for visualization
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    # Display the frame
    cv2.imshow("Pass 1: Analysis", frame)

    frame_index += 1
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Skipping ahead in analysis...")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Pass 1 finished. Analyzed {frame_index} frames.")

# Extract all unique track IDs found
all_track_ids = set()
for frame_data in all_frames_data.values():
    for track_info in frame_data:
        all_track_ids.add(track_info["id"])

if not all_track_ids:
    print("No tracks were detected in the video. Exiting.")
    exit()

print("\nAvailable Track IDs:", sorted(list(all_track_ids)))


# =================================================================================
# USER SELECTION
# =================================================================================
target_id = None
while target_id not in all_track_ids:
    try:
        selected_id = int(input("Enter the ID of the person you want to track: "))
        if selected_id in all_track_ids:
            target_id = selected_id
        else:
            print("Invalid ID. Please choose from the available IDs.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"Target ID {target_id} selected.")


# =================================================================================
# PASS 2: GENERATE OUTPUT VIDEO WITH TRAIL
# =================================================================================
print("Starting Pass 2: Generating output video with trail...")

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps_out = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (RESIZE_WIDTH, RESIZE_HEIGHT)
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps_out, frame_size)
trail_mask = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)
historical_points = []
frame_index = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    frame_tracks = all_frames_data.get(frame_index, [])
    
    for track_info in frame_tracks:
        if track_info["id"] == target_id:
            box = track_info["box"]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            leg_point = ((x1 + x2) // 2, y2)
            historical_points.append(leg_point)
            break

    if len(historical_points) > 1:
        for i in range(1, len(historical_points)):
            cv2.line(trail_mask, historical_points[i - 1], historical_points[i], (255, 0, 0), 2)

    final_frame = cv2.add(frame, trail_mask)
    out.write(final_frame)
    
    cv2.imshow("Pass 2: Generating Video...", final_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
         break
    frame_index += 1

print(f"\nProcessing finished. Output video saved as {OUTPUT_FILENAME}")
cap.release()
out.release()
cv2.destroyAllWindows()