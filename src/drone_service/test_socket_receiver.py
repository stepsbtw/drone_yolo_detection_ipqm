"""
Pure socket receiver test - NO library video input.
Just receives raw video from cellphone and displays it.
"""

import socket
import pickle
import struct
import cv2
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def receive_frames(port=5555):
    """Receive video frames via socket and display them."""
    
    logger.info("=" * 70)
    logger.info("PURE SOCKET RECEIVER TEST")
    logger.info("=" * 70)
    logger.info(f"Listening on port: {port}")
    logger.info("Waiting for cellphone connection...")
    logger.info("=" * 70)
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)
    
    try:
        # Wait for connection
        conn, addr = server_socket.accept()
        logger.info(f"✓ Connected to: {addr}")
        logger.info("=" * 70)
        logger.info("Receiving frames... Press 'q' to quit")
        logger.info("=" * 70)
        
        # Create window
        window_name = "Socket Receiver Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        data = b""
        payload_size = struct.calcsize("Q")
        frame_count = 0
        start_time = time.time()
        last_frame_time = time.time()
        
        while True:
            # Retrieve message size
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    logger.warning("Connection closed by client")
                    return
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Retrieve all data based on message size
            while len(data) < msg_size:
                data += conn.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # Deserialize frame
            frame = pickle.loads(frame_data)
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - start_time
            instant_fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            last_frame_time = current_time
            
            # Add info overlay
            cv2.putText(
                frame,
                f"Socket Test | Frame: {frame_count} | FPS: {instant_fps:.1f} | Avg: {avg_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Source: {addr[0]}:{addr[1]}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit requested")
                break
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 70)
        logger.info("STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total frames received: {frame_count}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        server_socket.close()
        logger.info("Socket closed")


if __name__ == "__main__":
    receive_frames(port=5555)
