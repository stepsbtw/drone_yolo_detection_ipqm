"""
Local viewer for testing drone service without REST server.
Displays annotated frames in real-time using OpenCV.
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
    """Main viewer loop."""
    
    logger.info("=" * 70)
    logger.info("DRONE DETECTOR - LOCAL VIEWER")
    logger.info("=" * 70)
    
    try:
        import drone_people_detector
        logger.info("✓ drone_people_detector library imported successfully")
    except ImportError as e:
        logger.error(f"drone_people_detector library import failed!")
        logger.error(f"ImportError details: {e}")
        logger.error(f"Full traceback:", exc_info=True)
        logger.error("Install with: pip install -e /home/caio.torkst/DRONE/drone_yolo_detection_ipqm")
        return 1
    
    # Initialize config
    config = Config()
    
    logger.info(f"Video source: {config.video.input_path}")
    logger.info(f"Camera ID: {config.camera.id}")
    logger.info(f"Person confidence: {config.detection.person_confidence}")
    logger.info(f"Weapon confidence: {config.detection.weapon_confidence}")
    logger.info(f"Weapon detection: {config.detection.enable_weapon_detection}")
    logger.info(f"Tracking: {config.detection.enable_tracking}")
    
    # Initialize detector
    logger.info("Initializing detector...")
    detector = drone_people_detector.DronePeopleDetector(config)
    camera = detector.get_camera()
    logger.info("✓ Detector initialized")
    
    # Wait for first frame
    logger.info("Waiting for first frame...")
    frame, tracks = detector.get_predict()
    if frame is None:
        logger.error("Failed to get first frame")
        return 1
    logger.info("✓ First frame received")
    
    logger.info("=" * 70)
    logger.info("Starting viewer... Press 'q' to quit, 'p' to pause")
    logger.info("=" * 70)
    
    # Create window
    window_name = f"Drone Detector - {config.camera.id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    frame_count = 0
    paused = False
    fps_limit = config.video.fps_limit if hasattr(config.video, 'fps_limit') else 10
    frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0.001
    
    start_time = time.time()
    
    try:
        while True:
            if not paused:
                # Get frame and predictions
                frame, tracks = detector.get_predict()
                
                if frame is None:
                    # End of video
                    if config.video.loop:
                        logger.info("Video ended, restarting...")
                        detector.release()
                        detector = drone_people_detector.DronePeopleDetector(config)
                        continue
                    else:
                        logger.info("Video ended")
                        break
                
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Draw bounding boxes
                # agora inclui contador de frames e FPS
                annotated_frame = detector.draw_bbox(frame, tracks, frame_count, fps)
                
                # Help text
                help_text = "Press 'q' to quit, 'p' to pause"
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
                filename = f"screenshot_{frame_count}.jpg"
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
