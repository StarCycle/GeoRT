import cv2
import numpy as np
import threading
import time
from MultiViewHandCapture.track import StereoHandTracker, HandVisualizerAllInOne
import matplotlib.pyplot as plt
import argparse
import geort

class MultiViewMocap:
    """
    Multi-view stereo hand tracking mocap system using MultiViewHandCapture library.
    Provides API compatible with GeoRT's mocap system.
    """

    def __init__(self):
        # Initialize the stereo hand tracker (uses camera ID from config)
        self.tracker = StereoHandTracker()
        self.status = 'idle'
        self.latest_result = None
        self.lock = threading.Lock()
        
        # Start tracking thread
        self.running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        print("MultiViewMocap initialized with stereo camera")

    def _tracking_loop(self):
        """Background thread for continuous hand tracking"""
        while self.running:
            try:
                # Get tracking data from stereo tracker
                result = self.tracker.step()
                
                with self.lock:
                    if result['found'] and result['keypoint_relative'] is not None:
                        self.latest_result = result['keypoint_relative']
                        self.status = 'recording'
                    else:
                        self.latest_result = None
                        self.status = 'no data'
                        
            except Exception as e:
                print(f"Tracking error: {e}")
                with self.lock:
                    self.latest_result = None
                    self.status = 'error'
                
            time.sleep(0.033)  # ~30 FPS

    def get(self):
        """
        Get latest hand tracking data.
        Returns: dict with 'status' and 'result' keys
        """
        with self.lock:
            if self.latest_result is not None:
                # Convert from mm to meters for GeoRT
                keypoint_3d = self.latest_result.astype(np.float32) * 0.001
                return {
                    'status': self.status,
                    'result': keypoint_3d.reshape(21, 3)
                }
            else:
                return {
                    'status': self.status,
                    'result': None
                }

    def close(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join(timeout=1.0)
        if hasattr(self, 'tracker'):
            self.tracker.close()
        print("MultiViewMocap closed")

    def __del__(self):
        self.close()

def run_demo_with_recording(dataset_name, max_frames=10000):
    """
    Demo using MultiViewHandCapture's built-in visualization system with data recording
    
    Args:
        dataset_name: Name for saving the dataset
        max_frames: Maximum number of frames to record (default: 5000)
    """
    print("Starting MultiView Hand Tracking Demo with Data Recording")
    print(f"Will automatically record {max_frames} frames")
    print("=" * 50)
    
    # Initialize tracker and visualizer
    tracker = StereoHandTracker()
    visualizer = HandVisualizerAllInOne()

    # Data collection variables
    all_results = []
    frame_count = 0
    collecting_data = False
    
    print("Starting calibration phase...")

    try:
        while frame_count < max_frames:
            # Process one frame
            data = tracker.step()
            
            if data["image_left"] is None:
                time.sleep(0.01)
                continue
            
            # Get current phase
            current_phase = data.get("phase", "unknown")
            
            # Update status display
            visualizer.set_status(data["phase"])

            # Update visualization with rotated dimensions
            rotated_width = data.get("rotated_width", data["image_left"].shape[1])
            rotated_height = data.get("rotated_height", data["image_left"].shape[0])

            # Check if we are in gesture capture phase
            if "GESTURE TRACKING" in current_phase:
                if not collecting_data:
                    print("\nGesture capture phase detected. Starting data collection...")
                    collecting_data = True
                
                # Record data if hand is detected
                if data['keypoint_relative'] is not None:
                    keypoint_3d = data['keypoint_relative'].astype(np.float64)
                    all_results.append(keypoint_3d.reshape(21, 3))
                    frame_count += 1
                    
                    # Print progress every 10 frames
                    if frame_count % 10 == 0:
                        print(f"Collected {frame_count}/{max_frames} frames. Press 'q' to quit.")
                
                if frame_count % 10 == 0:
                    visualizer.update(
                        data["image_left"], 
                        data["image_right"], 
                        data["px_left"], 
                        data["px_right"], 
                        data["keypoint_absolute"],
                        data["keypoint_relative"],
                        rotated_width,
                        rotated_height
                    )
            else:
                visualizer.update(
                    data["image_left"], 
                    data["image_right"], 
                    data["px_left"], 
                    data["px_right"], 
                    data["keypoint_absolute"],
                    data["keypoint_relative"],
                    rotated_width,
                    rotated_height
                )

            # Check if window was closed
            if not plt.fignum_exists(visualizer.fig.number):
                print("Visualization window closed. Stopping collection.")
                break

    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        tracker.close()
        plt.close('all')

        # Save collected data
        if len(all_results) > 0:
            save_path = geort.save_human_data(np.array(all_results), dataset_name)
            print("\n" + "=" * 50)
            print("Data collection complete!")
            print(f"Total frames collected: {len(all_results)}")
            print(f"Data saved to: {save_path}")
        else:
            print("No data collected.")

        print("Demo stopped.")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-view hand motion capture with data recording')
    parser.add_argument('-name', type=str, default='human_multi_view', 
                       help='Name of the dataset to save (default: human_multi_view)')
    parser.add_argument('-frames', type=int, default=10000,
                       help='Number of frames to record (default: 10000)')
    args = parser.parse_args()
    
    dataset_name = args.name
    max_frames = args.frames
    
    print(f"Dataset will be saved as: {dataset_name}")
    print(f"Will record {max_frames} frames")
    print("=" * 50)
    
    run_demo_with_recording(dataset_name, max_frames)