"""
Drone Service with Socket Video Input
Receives real-time video from cellphone connected to drone.
"""

import drone_people_detector
from injector import Injector
import time
import os
import sys
import logging
import cv2

from di import AppModule
from src.config import Config
from src.services.streamer import VideoStreamer
from src.services.broker.broker import Broker
from src.services.socket_video_capture import SocketVideoCapture

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Main service loop with socket video input."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("DRONE PEOPLE DETECTION SERVICE - SOCKET MODE")
    logger.info("=" * 70)
    
    # Initialize DI container
    injector = Injector([AppModule()])
    
    # Get dependencies via DI
    config = injector.get(Config)
    broker = injector.get(Broker)
    streamer = injector.get(VideoStreamer)
    
    # Check if socket mode is enabled
    use_socket = getattr(config.video, 'use_socket', False)
    socket_port = getattr(config.video, 'socket_port', 5555)
    
    # Initialize video capture
    socket_capture = None
    if use_socket:
        logger.info(f"📱 Socket mode enabled on port {socket_port}")
        logger.info("Waiting for cellphone connection...")
        
        # Create socket video capture
        socket_capture = SocketVideoCapture(host='0.0.0.0', port=socket_port)
        
        # Replace config video input with socket capture
        # We'll manually handle the detector initialization
        config.video.input_path = None  # Disable file-based input
    else:
        logger.info(f"📹 File mode: {config.video.input_path}")
    
    # Initialize detector
    logger.info("Initializing drone people detector...")
    detector = drone_people_detector.DronePeopleDetector(config)
    
    # If socket mode, replace the detector's cv2.VideoCapture with our socket capture
    if use_socket and socket_capture:
        detector.cap = socket_capture
        logger.info("✓ Socket video capture attached to detector")
    
    camera = detector.get_camera()
    logger.info(f"✓ Detector initialized")
    logger.info(f"Camera: {config.camera.id} at ({config.camera.latitude}, {config.camera.longitude})")
    
    # Start streamer
    streamer.start()
    logger.info("✓ Streamer started")
    
    # Wait for first frame
    if not wait_for_first_frame(detector, socket_capture, config.frame_timeout):
        logger.error("Timeout waiting for first frame")
        cleanup(detector, streamer, socket_capture)
        return 1
    
    logger.info("✓ Service initialized successfully")
    logger.info("=" * 70)
    logger.info("Starting main loop...")
    logger.info("=" * 70)
    
    frame_count = 0
    last_stats_time = time.time()
    
    # Main loop
    try:
        while True:
            # Get prediction from detector
            img_to_show, tracks_list = detector.get_predict()
            
            if img_to_show is None:
                if use_socket:
                    # Socket mode: wait for reconnection or new frames
                    logger.warning("No frame received, waiting...")
                    time.sleep(0.1)
                    continue
                else:
                    # File mode: handle end of video
                    if config.video.loop:
                        logger.info("Video ended, restarting...")
                        detector.release()
                        detector = drone_people_detector.DronePeopleDetector(config)
                        continue
                    else:
                        logger.info("Video ended, stopping service")
                        break
            
            # Draw bounding boxes and publish image
            annotated_frame = detector.draw_bbox(img_to_show, tracks_list)
            broker.write(annotated_frame)
            
            frame_count += 1
            
            # Log statistics periodically (every 3 seconds)
            current_time = time.time()
            if current_time - last_stats_time >= 3.0:
                people_count = len(tracks_list) if tracks_list else 0
                weapons_count = 0
                if tracks_list:
                    for t in tracks_list:
                        if hasattr(t, 'weapon_classifier') and t.weapon_classifier.has_weapon():
                            weapons_count += 1
                
                # Get connection status
                if use_socket and socket_capture:
                    stats = socket_capture.get_stats()
                    connection_status = f"📱 {'✓ Connected' if stats['connected'] else '✗ Waiting'} | FPS: {stats['fps']:.1f}"
                else:
                    connection_status = f"Stream: {'✓ Connected' if streamer.is_connected() else '✗ Disconnected'}"
                
                logger.info(
                    f"Frame {frame_count:5d} | {people_count:2d} people | "
                    f"{weapons_count:2d} weapons | {connection_status}"
                )
                
                last_stats_time = current_time
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received interrupt signal, stopping...")
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}", exc_info=True)
    
    # Cleanup
    cleanup(detector, streamer, socket_capture)
    
    logger.info("=" * 70)
    logger.info(f"✓ Service stopped. Processed {frame_count} frames")
    logger.info("=" * 70)
    
    return 0


def wait_for_first_frame(detector, socket_capture, timeout=30):
    """
    Wait for first frame from video source.
    
    Args:
        detector: DronePeopleDetector instance
        socket_capture: SocketVideoCapture instance (or None for file mode)
        timeout: Maximum time to wait (seconds)
    
    Returns:
        True if first frame received, False if timeout
    """
    logger = logging.getLogger(__name__)
    
    if socket_capture:
        logger.info(f"Waiting for first frame from cellphone (timeout: {timeout}s)...")
        logger.info("💡 Make sure the cellphone client is running and connected!")
    else:
        logger.info(f"Waiting for first frame from video file (timeout: {timeout}s)...")
    
    start_time = time.time()
    
    while True:
        img_to_show, tracks_list = detector.get_predict()
        if img_to_show is not None:
            logger.info("✅ First frame received!")
            return True
            
        if time.time() - start_time > timeout:
            logger.error(f"⏱️  Timeout after {timeout}s")
            return False
            
        time.sleep(0.1)


def cleanup(detector, streamer, socket_capture):
    """Cleanup resources."""
    logger = logging.getLogger(__name__)
    
    logger.info("Cleaning up resources...")
    
    try:
        detector.release()
    except:
        pass
        
    try:
        streamer.stop()
    except:
        pass
        
    try:
        if socket_capture:
            socket_capture.release()
    except:
        pass


if __name__ == "__main__":
    exit_code = main()
    os._exit(exit_code)
