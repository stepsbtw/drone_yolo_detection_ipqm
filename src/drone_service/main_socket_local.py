"""
Socket receiver with local OpenCV viewer (no streaming).
Receives video from cellphone via socket and displays with detection.
"""

import cv2
import time
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main viewer loop with socket input."""
    
    logger.info("=" * 70)
    logger.info("DRONE DETECTOR - SOCKET LOCAL VIEWER")
    logger.info("=" * 70)
    
    try:
        import drone_people_detector
        logger.info("✓ drone_people_detector library imported successfully")
    except ImportError as e:
        logger.error(f"drone_people_detector library import failed!")
        logger.error(f"ImportError details: {e}")
        logger.error("Install with: pip install -e /home/caio.torkst/DRONE/drone_yolo_detection_ipqm")
        return 1
    
    # Initialize config
    config = Config()
    
    # Override to use socket
    config.video.use_socket = True
    
    logger.info("=" * 70)
    logger.info("SOCKET CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Socket mode: {config.video.use_socket}")
    logger.info(f"Socket port: {config.video.socket_port}")
    logger.info(f"Camera ID: {config.camera.id}")
    logger.info(f"Person confidence: {config.detection.person_confidence}")
    logger.info(f"Weapon confidence: {config.detection.weapon_confidence}")
    logger.info(f"Weapon detection: {config.detection.enable_weapon_detection}")
    logger.info(f"Tracking: {config.detection.enable_tracking}")
    logger.info("=" * 70)
    
    # Initialize detector
    logger.info("Initializing detector with socket input...")
    logger.info(f"Waiting for connection on port {config.video.socket_port}...")
    logger.info("Send video from your cellphone now!")
    
    detector = drone_people_detector.DronePeopleDetector(config)
    camera = detector.get_camera()
    logger.info("✓ Detector initialized")
    
    # Wait for first frame
    logger.info("Waiting for first frame from socket...")
    frame, tracks = detector.get_predict()
    if frame is None:
        logger.error("Failed to get first frame from socket")
        logger.error("Make sure your cellphone is sending video!")
        return 1
    logger.info("✓ First frame received!")
    
    logger.info("=" * 70)
    logger.info("Starting viewer... Press 'q' to quit, 'p' to pause")
    logger.info("=" * 70)
    
    # Create window
    window_name = f"Drone Detector - Socket Input"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    frame_count = 0
    paused = False
    fps_limit = config.video.fps_limit if hasattr(config.video, 'fps_limit') else 10
    frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0.001
    
    start_time = time.time()
    last_frame_time = time.time()
    
    try:
        while True:
            if not paused:
                # Get frame and predictions
                frame, tracks = detector.get_predict()
                
                if frame is None:
                    logger.warning("Lost socket connection or no more frames")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Calculate FPS
                elapsed = current_time - start_time
                instant_fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                last_frame_time = current_time
                
                # Draw bounding boxes
                annotated_frame = detector.draw_bbox(frame, tracks, frame_count, avg_fps)
                
                # Add socket info
                socket_text = f"Socket Input | Instant FPS: {instant_fps:.1f}"
                cv2.putText(
                    annotated_frame,
                    socket_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Help text
                help_text = "Press 'q' to quit, 'p' to pause, 's' to screenshot"
                cv2.putText(
                    annotated_frame,
                    help_text,
                    (15, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Display frame
                cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('p'):
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                # Save current frame
                filename = f"screenshot_socket_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Screenshot saved: {filename}")
                
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        detector.release()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 70)
        logger.info("STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
