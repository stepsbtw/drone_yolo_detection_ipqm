#!/usr/bin/env python3
"""
Cellphone Video Client - Sends real-time video to drone_service

This Python script demonstrates the protocol for sending video frames.
It can be adapted for:
- Android (Java/Kotlin)
- iOS (Swift)
- Other platforms

Protocol:
1. Connect to server via TCP socket
2. For each frame:
   - Encode frame as JPEG
   - Send frame size as 4-byte unsigned int (network byte order)
   - Send JPEG data
3. Repeat

Usage:
    # Using webcam:
    python cellphone_client.py --server 192.168.1.100

    # Using video file:
    python cellphone_client.py --server 192.168.1.100 --source video.mp4
    
    # Adjust quality and FPS:
    python cellphone_client.py --server 192.168.1.100 --fps 15 --quality 85
"""

import socket
import cv2
import struct
import argparse
import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CellphoneVideoClient:
    """Client that sends video frames to drone_service server."""
    
    def __init__(self, server_host: str, server_port: int = 5555, 
                 jpeg_quality: int = 80, target_fps: int = 10):
        """
        Initialize cellphone video client.
        
        Args:
            server_host: Server IP address
            server_port: Server port number
            jpeg_quality: JPEG compression quality (0-100, higher = better quality)
            target_fps: Target frames per second to send
        """
        self.server_host = server_host
        self.server_port = server_port
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps if target_fps > 0 else 0
        
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to the drone_service server.
        
        Returns:
            True if connected successfully
        """
        try:
            logger.info(f"Connecting to {self.server_host}:{self.server_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            logger.info("✅ Connected to drone_service!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.connected = False
            logger.info("Disconnected from server")
            
    def send_frame(self, frame) -> bool:
        """
        Send a single frame to the server.
        
        Args:
            frame: Frame as numpy array (BGR format from OpenCV)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected:
            return False
            
        try:
            # Encode frame as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                logger.error("Failed to encode frame")
                return False
                
            # Get JPEG data as bytes
            jpeg_data = buffer.tobytes()
            data_size = len(jpeg_data)
            
            # Send frame size (4 bytes, network byte order = big-endian)
            size_bytes = struct.pack('!I', data_size)  # '!' = network byte order
            self.socket.sendall(size_bytes)
            
            # Send frame data
            self.socket.sendall(jpeg_data)
            
            return True
            
        except socket.error as e:
            logger.error(f"Socket error while sending: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            return False
            
    def stream_video(self, source):
        """
        Stream video from source to server.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video source: {source}")
            return
            
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info("=" * 60)
        logger.info(f"📹 Video source: {source}")
        logger.info(f"📐 Resolution: {width}x{height}")
        logger.info(f"🎬 Source FPS: {source_fps:.1f}")
        logger.info(f"📤 Target FPS: {self.target_fps}")
        logger.info(f"🖼️  JPEG quality: {self.jpeg_quality}%")
        logger.info("=" * 60)
        logger.info("📱 Streaming... Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        frame_count = 0
        bytes_sent = 0
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video or camera disconnected")
                    break
                
                # Send frame
                if not self.send_frame(frame):
                    logger.error("Failed to send frame, stopping")
                    break
                    
                frame_count += 1
                
                # Estimate bytes sent (approximate)
                # Actual size varies with JPEG compression
                bytes_sent += width * height * 3 * self.jpeg_quality // 100
                
                # Log statistics every 3 seconds
                current_time = time.time()
                if current_time - last_log_time >= 3.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    mbps = (bytes_sent * 8 / 1000000) / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"📊 Frames: {frame_count:5d} | "
                        f"FPS: {fps:5.1f} | "
                        f"Bandwidth: {mbps:5.2f} Mbps"
                    )
                    last_log_time = current_time
                    
                # FPS limiting (sleep to maintain target FPS)
                elapsed_frame = time.time() - frame_start
                sleep_time = self.frame_delay - elapsed_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")
        finally:
            cap.release()
            
            # Final statistics
            total_time = time.time() - start_time
            if total_time > 0:
                final_fps = frame_count / total_time
                final_mbps = (bytes_sent * 8 / 1000000) / total_time
                
                logger.info("=" * 60)
                logger.info("📊 Final Statistics:")
                logger.info(f"  Total frames: {frame_count}")
                logger.info(f"  Duration: {total_time:.1f}s")
                logger.info(f"  Average FPS: {final_fps:.1f}")
                logger.info(f"  Average bandwidth: {final_mbps:.2f} Mbps")
                logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cellphone Video Client for drone_service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream from webcam to server at 192.168.1.100
  python cellphone_client.py --server 192.168.1.100

  # Stream from video file
  python cellphone_client.py --server 192.168.1.100 --source video.mp4

  # Higher quality and FPS
  python cellphone_client.py --server 192.168.1.100 --fps 20 --quality 90
        """
    )
    
    parser.add_argument('--server', required=True, 
                       help='Server IP address (drone_service IP)')
    parser.add_argument('--port', type=int, default=5555, 
                       help='Server port (default: 5555)')
    parser.add_argument('--source', default='0', 
                       help='Video source: 0 for webcam, or path to video file (default: 0)')
    parser.add_argument('--fps', type=int, default=10, 
                       help='Target FPS to send (default: 10)')
    parser.add_argument('--quality', type=int, default=80, 
                       help='JPEG quality 0-100 (default: 80, higher = better quality)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.quality < 0 or args.quality > 100:
        logger.error("Quality must be between 0 and 100")
        return 1
        
    if args.fps < 1 or args.fps > 60:
        logger.error("FPS must be between 1 and 60")
        return 1
    
    # Convert source to int if it's a digit (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create client
    client = CellphoneVideoClient(
        server_host=args.server,
        server_port=args.port,
        jpeg_quality=args.quality,
        target_fps=args.fps
    )
    
    # Connect to server
    if not client.connect():
        return 1
        
    try:
        # Stream video
        client.stream_video(source)
    finally:
        # Cleanup
        client.disconnect()
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
