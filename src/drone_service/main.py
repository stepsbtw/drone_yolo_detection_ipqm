import drone_people_detector
from injector import Injector
import time
import os
import sys
import logging

from di import AppModule
from src.config import Config
from src.services.streamer import VideoStreamer
from src.services.broker.broker import Broker
#import logging_config as logger_config

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def main():
    """Main service loop."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("DRONE PEOPLE DETECTION SERVICE")
    logger.info("=" * 70)
    
    # Initialize DI container
    injector = Injector([AppModule()])
    
    # Get dependencies via DI
    config = injector.get(Config)
    broker = injector.get(Broker)
    streamer = injector.get(VideoStreamer)
    
    # Initialize detector (uses config)
    logger.info("Initializing drone people detector...")
    drone_detector = drone_people_detector.DronePeopleDetector(config)
    camera = drone_detector.get_camera()
    logger.info(f"✓ Detector initialized")
    logger.info(f"Camera: {config.camera.id} at ({config.camera.latitude}, {config.camera.longitude})")
    
    # Start streamer
    streamer.start()
    logger.info("✓ Streamer started")
    
    # Wait for first frame (keyframe)
    if not wait_for_keyframe(drone_detector, config.frame_timeout):
        logger.error("Timeout waiting for first frame")
        return 1
    
    logger.info("✓ Service initialized successfully")
    logger.info("=" * 70)
    logger.info("Starting main loop...")
    logger.info("=" * 70)
    
    frame_count = 0
    
    # Main loop
    while True:
        try:
            # Get prediction from detector
            img_to_show, tracks_list = drone_detector.get_predict()
            
            if img_to_show is None:
                # End of video
                if config.video.loop:
                    logger.info("Video ended, restarting...")
                    # Reinitialize detector to restart video
                    drone_detector = drone_people_detector.DronePeopleDetector(config)
                    continue
                else:
                    logger.info("Video ended, stopping service")
                    break
            
            # Draw bounding boxes and publish image
            annotated_frame = drone_detector.draw_bbox(img_to_show, tracks_list)
            broker.write(annotated_frame)
            
            frame_count += 1
            
            # Log statistics periodically
            if frame_count % 30 == 0:
                people_count = len(tracks_list) if tracks_list else 0
                # Conta armas corretamente - PersonTrack objects
                weapons_count = 0
                if tracks_list:
                    for t in tracks_list:
                        if hasattr(t, 'weapon_classifier') and t.weapon_classifier.has_weapon():
                            weapons_count += 1
                logger.info(
                    f"Frame {frame_count:4d}: {people_count:2d} people, "
                    f"{weapons_count:2d} with weapons | "
                    f"Stream: {'✓ Connected' if streamer.is_connected() else '✗ Disconnected'}"
                )
            
            # FPS limiting
            time.sleep(0.01)
            
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal, stopping...")
            break
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(0.1)
    
    # Cleanup
    drone_detector.release()
    streamer.stop()
    logger.info("=" * 70)
    logger.info(f"Service stopped. Processed {frame_count} frames")
    logger.info("=" * 70)
    
    return 0


def wait_for_keyframe(drone_detector, timeout=10):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    logger.info(f"Waiting for first frame (timeout: {timeout}s)...")
    
    while True:
        img_to_show, tracks_list = drone_detector.get_predict()
        if img_to_show is not None:
            logger.info("✓ First frame received")
            return True
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)


if __name__ == "__main__":
    exit_code = main()
    os._exit(exit_code)
