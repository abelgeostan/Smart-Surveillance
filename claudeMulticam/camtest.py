import cv2
import numpy as np
import threading
import time

class DualCameraViewer:
    def __init__(self, url1, url2):
        self.url1 = url1
        self.url2 = url2
        self.cap1 = None
        self.cap2 = None
        self.frame1 = None
        self.frame2 = None
        self.running = False
        self.cam1_status = "Connecting..."
        self.cam2_status = "Connecting..."
        
    def connect_cameras(self):
        """Connect to both cameras"""
        print("Connecting to cameras...")
        
        # Connect to camera 1
        self.cap1 = cv2.VideoCapture(self.url1)
        self.cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if ret:
                self.cam1_status = f"✅ Connected - {frame.shape}"
                print(f"Camera 1: {self.cam1_status}")
            else:
                self.cam1_status = "❌ No frames"
                print(f"Camera 1: {self.cam1_status}")
        else:
            self.cam1_status = "❌ Connection failed"
            print(f"Camera 1: {self.cam1_status}")
        
        # Connect to camera 2
        self.cap2 = cv2.VideoCapture(self.url2)
        self.cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if self.cap2.isOpened():
            ret, frame = self.cap2.read()
            if ret:
                self.cam2_status = f"✅ Connected - {frame.shape}"
                print(f"Camera 2: {self.cam2_status}")
            else:
                self.cam2_status = "❌ No frames"
                print(f"Camera 2: {self.cam2_status}")
        else:
            self.cam2_status = "❌ Connection failed"
            print(f"Camera 2: {self.cam2_status}")
    
    def capture_camera1(self):
        """Capture frames from camera 1 in separate thread"""
        while self.running:
            if self.cap1 and self.cap1.isOpened():
                ret, frame = self.cap1.read()
                if ret:
                    self.frame1 = frame
                else:
                    self.frame1 = None
            time.sleep(0.03)  # ~30 FPS
    
    def capture_camera2(self):
        """Capture frames from camera 2 in separate thread"""
        while self.running:
            if self.cap2 and self.cap2.isOpened():
                ret, frame = self.cap2.read()
                if ret:
                    self.frame2 = frame
                else:
                    self.frame2 = None
            time.sleep(0.03)  # ~30 FPS
    
    def run(self):
        """Main display loop"""
        self.connect_cameras()
        
        # Start capture threads
        self.running = True
        thread1 = threading.Thread(target=self.capture_camera1)
        thread2 = threading.Thread(target=self.capture_camera2)
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        
        print("\nDisplaying camera feeds...")
        print("Press 'q' to quit")
        print("Press 'r' to reconnect cameras")
        print("Press 's' to save current frames")
        
        frame_count = 0
        prev_time = 0
        
        while True:
            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Display camera 1
            if self.frame1 is not None:
                # Add status text to frame
                display_frame1 = self.frame1.copy()
                cv2.putText(display_frame1, f"Camera 1: {self.url1}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame1, f"FPS: {fps:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Camera 1', display_frame1)
            else:
                # Show placeholder if no frame
                placeholder1 = cv2.imread('placeholder.jpg') if cv2.haveImageReader('placeholder.jpg') else \
                              (128 * np.ones((480, 640, 3), dtype='uint8'))
                cv2.putText(placeholder1, f"Camera 1: {self.cam1_status}", 
                           (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Camera 1', placeholder1)
            
            # Display camera 2
            if self.frame2 is not None:
                # Add status text to frame
                display_frame2 = self.frame2.copy()
                cv2.putText(display_frame2, f"Camera 2: {self.url2}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame2, f"FPS: {fps:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Camera 2', display_frame2)
            else:
                # Show placeholder if no frame
                placeholder2 = 128 * np.ones((480, 640, 3), dtype='uint8')
                cv2.putText(placeholder2, f"Camera 2: {self.cam2_status}", 
                           (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Camera 2', placeholder2)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Reconnecting cameras...")
                self.cleanup_cameras()
                self.connect_cameras()
            elif key == ord('s'):
                # Save current frames
                if self.frame1 is not None:
                    cv2.imwrite(f'camera1_frame_{frame_count}.jpg', self.frame1)
                    print(f"Saved camera 1 frame {frame_count}")
                if self.frame2 is not None:
                    cv2.imwrite(f'camera2_frame_{frame_count}.jpg', self.frame2)
                    print(f"Saved camera 2 frame {frame_count}")
            
            frame_count += 1
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Frames processed: {frame_count}, FPS: {fps:.2f}")
        
        # Cleanup
        self.running = False
        self.cleanup_cameras()
        cv2.destroyAllWindows()
        print("Cameras disconnected and windows closed.")
    
    def cleanup_cameras(self):
        """Clean up camera resources"""
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()

# Usage
if __name__ == "__main__":
    # Your camera URLs
    camera_urls = [
        'http://10.39.128.194:4747/video',
        'http://10.39.128.141:4747/video'
    ]
    
    # Create and run viewer
    viewer = DualCameraViewer(camera_urls[0], camera_urls[1])
    
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        viewer.cleanup_cameras()
        cv2.destroyAllWindows()