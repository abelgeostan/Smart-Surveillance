import cv2
import numpy as np
import json
import os

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CCTVStitcher:
    def __init__(self, camera1_source, camera2_source, calibration_file='cctv_calibration.json'):
        """
        Initialize CCTV Stitcher
        camera_source can be:
        - Camera ID (int): 0, 1, 2, etc.
        - RTSP URL (str): 'rtsp://username:password@ip:port/stream'
        - Video file (str): 'path/to/video.mp4'
        """
        self.camera1_source = camera1_source
        self.camera2_source = camera2_source
        self.calibration_file = calibration_file
        
        # Initialize video captures
        self.cap1 = cv2.VideoCapture(camera1_source)
        self.cap2 = cv2.VideoCapture(camera2_source)
        
        # Set buffer size to reduce latency for live feeds
        self.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Calibration data
        self.homography = None
        self.canvas_size = None
        self.offset = None
        self.is_calibrated = False
        
        # Try to load existing calibration
        self.load_calibration()
        
        # Feature detector for calibration
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def capture_sample_frames(self, num_frames=5):
        """Capture multiple sample frames for robust calibration"""
        print("Capturing sample frames for calibration...")
        
        all_good_matches = []
        frames_captured = 0
        
        while frames_captured < num_frames:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print("Error: Could not read from cameras")
                return None
            
            # Show current frames
            display = np.hstack([frame1, frame2])
            cv2.imshow('Calibration Frames - Press SPACE to capture, Q to quit', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar to capture
                matches = self.find_matches(frame1, frame2)
                if matches and len(matches) > 20:
                    all_good_matches.extend(matches)
                    frames_captured += 1
                    print(f"Frame {frames_captured}/{num_frames} captured with {len(matches)} matches")
                else:
                    print("Not enough matches in this frame, try adjusting camera angles")
            elif key == ord('q'):
                break
        
        cv2.destroyWindow('Calibration Frames - Press SPACE to capture, Q to quit')
        return all_good_matches if len(all_good_matches) > 50 else None
    
    def find_matches(self, img1, img2):
        """Find feature matches between two images"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None
        
        # Match descriptors
        matches = self.matcher.match(des1, des2)
        
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter good matches (top 30%)
        num_good_matches = int(len(matches) * 0.3)
        good_matches = matches[:num_good_matches]
        
        # Extract coordinate pairs
        match_pairs = []
        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            match_pairs.append((pt1, pt2))
        
        return match_pairs
    
    def calculate_stitching_parameters(self, matches):
        """Calculate homography and canvas parameters from matches"""
        # Prepare points for homography calculation
        src_pts = np.float32([match[0] for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1] for match in matches]).reshape(-1, 1, 2)
        
        # Calculate homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.99
        )
        
        if homography is None:
            return None
        
        # Get sample frame to calculate canvas size
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            return None
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Calculate canvas size and offset
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        # Transform corners of first image
        warped_corners1 = cv2.perspectiveTransform(corners1, homography)
        
        # Find bounding rectangle
        all_corners = np.concatenate([corners2, warped_corners1])
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        # Canvas size and offset
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        offset = (x_min, y_min)
        
        # Translation matrix to handle negative coordinates
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Final homography including translation
        final_homography = np.dot(translation_matrix, homography)
        
        return {
            'homography': final_homography.tolist(),
            'canvas_size': (canvas_width, canvas_height),
            'offset': offset,
            'camera2_position': (-x_min, -y_min)  # Where to place camera 2 on canvas
        }
    
    def calibrate(self):
        """Perform one-time calibration"""
        print("Starting CCTV calibration...")
        print("Make sure both cameras have overlapping view of the same scene")
        
        # Capture sample frames
        matches = self.capture_sample_frames()
        if not matches:
            print("Calibration failed: Not enough matches found")
            return False
        
        # Calculate stitching parameters
        params = self.calculate_stitching_parameters(matches)
        if not params:
            print("Calibration failed: Could not calculate homography")
            return False
        
        # Store calibration data
        self.homography = np.array(params['homography'], dtype=np.float32)
        self.canvas_size = params['canvas_size']
        self.offset = params['offset']
        self.camera2_position = params['camera2_position']
        self.is_calibrated = True
        
        # Save calibration
        self.save_calibration()
        print("Calibration completed successfully!")
        return True
    
    def save_calibration(self):
        """Save calibration data to file"""
        calibration_data = {
            'homography': self.homography.tolist(),
            'canvas_size': self.canvas_size,
            'offset': self.offset,
            'camera2_position': self.camera2_position,
            'camera1_source': str(self.camera1_source),
            'camera2_source': str(self.camera2_source)
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2, cls=NumpyEncoder) # Use the custom encoder here
        print(f"Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """Load calibration data from file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                
                self.homography = np.array(data['homography'], dtype=np.float32)
                self.canvas_size = tuple(data['canvas_size'])
                self.offset = tuple(data['offset'])
                self.camera2_position = tuple(data['camera2_position'])
                self.is_calibrated = True
                print(f"Calibration loaded from {self.calibration_file}")
                return True
            except:
                print("Failed to load calibration file")
                return False
        return False
    
    def stitch_frame(self, frame1, frame2):
        """Stitch two frames using pre-calculated parameters"""
        if not self.is_calibrated:
            print("Error: Not calibrated. Run calibrate() first.")
            return None
        
        # Create canvas
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        # Warp and place first camera frame
        warped_frame1 = cv2.warpPerspective(frame1, self.homography, self.canvas_size)
        
        # Place second camera frame
        x_start, y_start = self.camera2_position
        h2, w2 = frame2.shape[:2]
        
        # Ensure we don't go out of canvas bounds
        x_end = min(x_start + w2, self.canvas_size[0])
        y_end = min(y_start + h2, self.canvas_size[1])
        
        if x_start >= 0 and y_start >= 0:
            canvas[y_start:y_end, x_start:x_end] = frame2[:y_end-y_start, :x_end-x_start]
        
        # Handle overlapping regions with simple blending
        mask1 = (warped_frame1.sum(axis=2) > 0)
        mask2 = (canvas.sum(axis=2) > 0)
        overlap = mask1 & mask2
        
        # Blend overlapping areas
        result = canvas.copy()
        result[overlap] = (warped_frame1[overlap].astype(np.float32) * 0.5 + 
                          canvas[overlap].astype(np.float32) * 0.5).astype(np.uint8)
        
        # Add non-overlapping areas from camera 1
        result[mask1 & ~mask2] = warped_frame1[mask1 & ~mask2]
        
        return result
    
    def run_stitched_feed(self):
        """Run the stitched feed (main loop for your YOLO integration)"""
        if not self.is_calibrated:
            print("Error: Not calibrated. Run calibrate() first.")
            return
        
        print("Starting stitched feed...")
        print("Press 'q' to quit, 's' to save frame, 'r' to recalibrate")
        
        frame_count = 0
        
        while True:
            # Read frames
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading frames")
                break
            
            # Stitch frames
            stitched = self.stitch_frame(frame1, frame2)
            
            if stitched is not None:
                # Display result (remove this when integrating with YOLO)
                cv2.imshow('Stitched CCTV Feed', stitched)
                
                # This is where you would pass 'stitched' to your YOLO model
                # yolo_results = your_yolo_model(stitched)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Processed {frame_count} frames")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and stitched is not None:
                cv2.imwrite(f'stitched_frame_{frame_count}.jpg', stitched)
                print(f"Saved frame {frame_count}")
            elif key == ord('r'):
                print("Recalibrating...")
                if self.calibrate():
                    print("Recalibration successful")
    
    def get_stitched_frame(self):
        """Get a single stitched frame (useful for integration with other systems)"""
        if not self.is_calibrated:
            return None
        
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if ret1 and ret2:
            return self.stitch_frame(frame1, frame2)
        return None
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cap1'):
            self.cap1.release()
        if hasattr(self, 'cap2'):
            self.cap2.release()
        cv2.destroyAllWindows()

# Usage Example
if __name__ == "__main__":
    # Initialize with camera sources
    # For USB cameras: use integers (0, 1, 2, etc.)
    # For IP cameras: use RTSP URLs like 'rtsp://admin:password@192.168.1.100:554/stream'
    # For HTTP streams: use full HTTP URLs
    
    stitcher = CCTVStitcher(
        camera1_source='http://10.39.128.141:4747/video',  # First IP camera
        camera2_source='http://10.39.128.194:4747/video',  # Second IP camera
        calibration_file='ip_camera_setup.json'
    )
    
    try:
        # If not already calibrated, run calibration
        if not stitcher.is_calibrated:
            print("No existing calibration found. Starting calibration...")
            if not stitcher.calibrate():
                print("Calibration failed!")
                exit(1)
        
        # Run the stitched feed
        stitcher.run_stitched_feed()
        
        # Alternative: Get single frames for YOLO integration
        # while True:
        #     stitched_frame = stitcher.get_stitched_frame()
        #     if stitched_frame is not None:
        #         # Process with YOLO
        #         # results = yolo_model(stitched_frame)
        #         pass
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        del stitcher