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

class ManualCCTVStitcher:
    def __init__(self, camera1_source, camera2_source, calibration_file='manual_cctv_calibration.json'):
        """
        Initialize Manual CCTV Stitcher
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
        
        # Manual point selection
        self.reference_points = []  # Points from camera 1
        self.target_points = []     # Corresponding points from camera 2
        self.current_image = None
        self.point_selection_mode = None  # 'reference' or 'target'
        
        # Try to load existing calibration
        self.load_calibration()
        
    def select_points_manually(self):
        """Manual point selection interface for precise calibration"""
        print("\n=== MANUAL POINT SELECTION ===")
        print("Instructions:")
        print("1. First, select 4+ points from Camera 1 (reference)")
        print("2. Then select the same points from Camera 2 (target)")
        print("3. Points should be distinctive features visible in both cameras")
        print("4. Press 'SPACE' to capture frames, 'r' to reset, 'q' to quit")
        
        # Capture current frames
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            print("Error: Could not read from cameras")
            return False
        
        # Reset points
        self.reference_points = []
        self.target_points = []
        
        # Step 1: Select points from Camera 1
        print("\nStep 1: Select points from Camera 1 (Reference)")
        print("Click on distinctive points, then press SPACE when done")
        self.point_selection_mode = 'reference'
        self.current_image = frame1.copy()
        self._point_selection_loop("Camera 1 - Select Reference Points (Press SPACE when done)")
        
        if len(self.reference_points) < 4:
            print("Error: Need at least 4 points for homography calculation")
            return False
        
        # Step 2: Select corresponding points from Camera 2
        print("\nStep 2: Select corresponding points from Camera 2 (Target)")
        print(f"Select {len(self.reference_points)} points in the same order")
        self.point_selection_mode = 'target'
        self.current_image = frame2.copy()
        self._point_selection_loop("Camera 2 - Select Target Points (Press SPACE when done)")
        
        if len(self.target_points) != len(self.reference_points):
            print("Error: Number of target points must match reference points")
            return False
        
        print(f"Successfully selected {len(self.reference_points)} point pairs")
        return True
    
    def _point_selection_loop(self, window_name):
        """Internal method for point selection loop"""
        temp_image = self.current_image.copy()
        points = self.reference_points if self.point_selection_mode == 'reference' else self.target_points
        
        # Create window and set mouse callback
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        while True:
            # Display image with current points
            display_image = temp_image.copy()
            
            # Draw existing points
            for i, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(display_image, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(display_image, str(i+1), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw instructions
            cv2.putText(display_image, f"Points selected: {len(points)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Click to add point, 'r' to reset, SPACE to continue", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to continue
                break
            elif key == ord('r'):  # Reset points
                if self.point_selection_mode == 'reference':
                    self.reference_points = []
                else:
                    self.target_points = []
                points = self.reference_points if self.point_selection_mode == 'reference' else self.target_points
                temp_image = self.current_image.copy()
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return False
        
        cv2.destroyWindow(window_name)
        return True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point_selection_mode == 'reference':
                self.reference_points.append((x, y))
                print(f"Reference point {len(self.reference_points)}: ({x}, {y})")
            else:
                self.target_points.append((x, y))
                print(f"Target point {len(self.target_points)}: ({x}, {y})")
    
    def calculate_homography_from_points(self):
        """Calculate homography from manually selected points"""
        if len(self.reference_points) < 4 or len(self.target_points) < 4:
            print("Error: Need at least 4 point pairs")
            return None
        
        # Convert to numpy arrays
        src_pts = np.float32(self.reference_points).reshape(-1, 1, 2)
        dst_pts = np.float32(self.target_points).reshape(-1, 1, 2)
        
        # Calculate homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        if homography is None:
            print("Error: Could not calculate homography")
            return None
        
        # Calculate reprojection error
        reprojected_pts = cv2.perspectiveTransform(src_pts, homography)
        error = np.mean(np.sqrt(np.sum((reprojected_pts - dst_pts) ** 2, axis=2)))
        print(f"Reprojection error: {error:.2f} pixels")
        
        if error > 10.0:
            print("Warning: High reprojection error. Consider reselecting points.")
        
        return homography
    
    def calculate_canvas_parameters(self, homography):
        """Calculate canvas size and positioning"""
        # Get frame dimensions
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            return None
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Calculate canvas size
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        # Transform corners of first image
        warped_corners1 = cv2.perspectiveTransform(corners1, homography)
        
        # Find bounding rectangle
        all_corners = np.concatenate([corners2, warped_corners1])
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        offset = (x_min, y_min)
        
        # Translation matrix
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Final homography including translation
        final_homography = np.dot(translation_matrix, homography)
        
        return {
            'homography': final_homography,
            'canvas_size': (canvas_width, canvas_height),
            'offset': offset,
            'camera2_position': (-x_min, -y_min)
        }
    
    def preview_stitching(self):
        """Show stitching preview with manual points"""
        if len(self.reference_points) < 4 or len(self.target_points) < 4:
            print("Need at least 4 point pairs for preview")
            return
        
        homography = self.calculate_homography_from_points()
        if homography is None:
            return
        
        params = self.calculate_canvas_parameters(homography)
        if not params:
            return
        
        # Create preview
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            return
        
        # Warp frame1
        warped = cv2.warpPerspective(frame1, params['homography'], params['canvas_size'])
        
        # Place frame2
        x_start, y_start = params['camera2_position']
        h2, w2 = frame2.shape[:2]
        x_end = min(x_start + w2, params['canvas_size'][0])
        y_end = min(y_start + h2, params['canvas_size'][1])
        
        result = warped.copy()
        if x_start >= 0 and y_start >= 0:
            result[y_start:y_end, x_start:x_end] = frame2[:y_end-y_start, :x_end-x_start]
        
        # Show preview
        cv2.imshow('Stitching Preview (Press any key to continue)', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return params
    
    def calibrate_manual(self):
        """Manual calibration process"""
        print("=== MANUAL CALIBRATION ===")
        
        # Step 1: Point selection
        if not self.select_points_manually():
            print("Point selection failed")
            return False
        
        # Step 2: Preview
        print("\nShowing stitching preview...")
        params = self.preview_stitching()
        if not params:
            print("Preview failed")
            return False
        
        # Step 3: Confirm
        print("\nDoes the stitching look good?")
        print("Press 'y' to accept, 'n' to reselect points")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('y'):
            # Save calibration
            self.homography = params['homography']
            self.canvas_size = params['canvas_size']
            self.offset = params['offset']
            self.camera2_position = params['camera2_position']
            self.is_calibrated = True
            
            self.save_calibration()
            print("Calibration saved successfully!")
            return True
        else:
            print("Restarting calibration...")
            return self.calibrate_manual()
    
    def save_calibration(self):
        """Save calibration data"""
        calibration_data = {
            'homography': self.homography.tolist(),
            'canvas_size': self.canvas_size,
            'offset': self.offset,
            'camera2_position': self.camera2_position,
            'reference_points': self.reference_points,
            'target_points': self.target_points
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2, cls=NumpyEncoder)
        print(f"Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """Load calibration data"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                
                self.homography = np.array(data['homography'])
                self.canvas_size = tuple(data['canvas_size'])
                self.offset = tuple(data['offset'])
                self.camera2_position = tuple(data['camera2_position'])
                self.reference_points = data.get('reference_points', [])
                self.target_points = data.get('target_points', [])
                self.is_calibrated = True
                print(f"Calibration loaded from {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Failed to load calibration: {e}")
        return False

    # The stitch_frame, run_stitched_feed, and get_stitched_frame methods 
    # remain the same as in the original CCTVStitcher class
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
        """Run the stitched feed"""
        if not self.is_calibrated:
            print("Error: Not calibrated. Run calibrate() first.")
            return
        
        print("Starting stitched feed...")
        print("Press 'q' to quit, 's' to save frame, 'r' to recalibrate")
        
        frame_count = 0
        
        while True:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading frames")
                break
            
            stitched = self.stitch_frame(frame1, frame2)
            
            if stitched is not None:
                cv2.imshow('Stitched CCTV Feed', stitched)
                frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and stitched is not None:
                cv2.imwrite(f'stitched_frame_{frame_count}.jpg', stitched)
                print(f"Saved frame {frame_count}")
            elif key == ord('r'):
                print("Recalibrating...")
                if self.calibrate_manual():
                    print("Recalibration successful")

    def get_stitched_frame(self):
        """Get a single stitched frame"""
        if not self.is_calibrated:
            return None
        
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if ret1 and ret2:
            return self.stitch_frame(frame1, frame2)
        return None

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap1'):
            self.cap1.release()
        if hasattr(self, 'cap2'):
            self.cap2.release()
        cv2.destroyAllWindows()

# Usage Example
if __name__ == "__main__":
    stitcher = ManualCCTVStitcher(
        camera1_source='http://10.39.128.141:4747/video',  # First IP camera
        camera2_source='http://10.39.128.194:4747/video',  # Second IP camera
        calibration_file='manual_calibration.json'
    )
    
    try:
        if not stitcher.is_calibrated:
            print("No calibration found. Starting manual calibration...")
            if stitcher.calibrate_manual():
                print("Manual calibration successful!")
            else:
                print("Manual calibration failed!")
                exit(1)
        
        # Run stitched feed
        stitcher.run_stitched_feed()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        del stitcher