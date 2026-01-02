"""
Multi-view stereo camera evaluation script for hand retargeting.
This script validates robot control in simulation using stereo camera input.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from geort.env.hand import HandKinematicModel
from geort import load_model, get_config
from MultiViewHandCapture.track import StereoHandTracker, HandVisualizerAllInOne

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-view stereo evaluation for hand retargeting")
    parser.add_argument("-ckpt_tag", type=str, required=True, help="Checkpoint tag for the retargeting model")
    parser.add_argument("-hand", type=str, default="leap", help="Robot hand type")
    parser.add_argument("-enable_control", action="store_true", help="Enable position control mode")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 50)
    print("Multi-View Stereo Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.ckpt_tag}")
    print(f"Robot hand: {args.hand}")
    print(f"Control mode: {'Position Control' if args.enable_control else 'Direct Set'}")
    print("=" * 50)
    
    # Load GeoRT retargeting model
    model = load_model(args.ckpt_tag)
    print(f"[INFO] Loaded retargeting model: {args.ckpt_tag}")
    
    # Initialize stereo tracker and visualizer from MultiViewHandCapture
    tracker = StereoHandTracker()
    visualizer = HandVisualizerAllInOne()
    print("[INFO] Initialized stereo camera tracker and visualizer")
    
    # Initialize robot hand simulation
    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()
    print(f"[INFO] Initialized robot hand simulation: {args.hand}")
    
    print("\n[INFO] Starting evaluation loop...")
    print("[INFO] Close the visualization window or press Ctrl+C to quit")
    
    # Main loop
    try:
        while True:
            # Update simulation viewer
            if args.enable_control:
                viewer_env.update()
            else:
                viewer_env.update_render_only()
            
            # Get tracking data from stereo tracker
            data = tracker.step()
            
            if data["image_left"] is None:
                continue
            
            # Update visualizer status
            visualizer.set_status(data["phase"])
            
            # Get rotated dimensions for visualization
            rotated_width = data.get("rotated_width", data["image_left"].shape[1])
            rotated_height = data.get("rotated_height", data["image_left"].shape[0])
            
            # Update visualization using MultiViewHandCapture's API
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
            
            # Process tracking result for robot control
            if data['keypoint_relative'] is not None:
                keypoint_3d = data['keypoint_relative'].astype(np.float64).reshape(21, 3)
                
                # Forward through retargeting model
                qpos = model.forward(keypoint_3d)
                
                # Apply to robot hand
                if args.enable_control:
                    hand.set_qpos_target(qpos)
                else:
                    hand.set_qpos(qpos)
            
            # Check if visualization window was closed
            if not plt.fignum_exists(visualizer.fig.number):
                print("[INFO] Visualization window closed")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        tracker.close()
        plt.close('all')
        print("[INFO] Evaluation finished")

if __name__ == '__main__':
    main()