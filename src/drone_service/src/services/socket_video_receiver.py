"""
Socket-based video receiver for real-time drone video streaming.
Receives video frames from a cellphone connected to the drone.
"""

import socket
import threading
import queue
import logging
import struct
import numpy as np
import cv2
from typing import Optional
import time

logger = logging.getLogger(__name__)


class SocketVideoReceiver:
    """
    TCP Socket server that receives real-time video frames from a cellphone.
    
    Protocol (simple and efficient):
    1. Client sends frame size as 4-byte unsigned integer (network byte order)
    2. Client sends JPEG-encoded frame data
    3. Repeat for each frame
    
    The server buffers frames in a queue for consumption by the detector.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5555, max_buffer: int = 5):
        """
        Initialize socket video receiver.
        
        Args:
            host: Host address to bind (0.0.0.0 = all interfaces)
            port: Port number to listen on
            max_buffer: Maximum frames to buffer (old frames dropped if full)
        """
        self.host = host
        self.port = port
        self.max_buffer = max_buffer
        
        # Frame buffer (thread-safe queue)
        self.frame_queue = queue.Queue(maxsize=max_buffer)
        
        # Socket management
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        
        # Threading
        self.running = False
        self.server_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frames_received = 0
        self.bytes_received = 0
        self.last_frame_time = 0
        self.fps = 0.0
        
    def start(self):
        """Start the socket server in background thread."""
        if self.running:
            logger.warning("Socket receiver already running")
            return
            
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        logger.info(f"🔌 Socket video receiver started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the socket server and cleanup resources."""
        logger.info("Stopping socket video receiver...")
        self.running = False
        
        # Close connections
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        # Wait for thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
            
        logger.info("Socket video receiver stopped")
        
    def _server_loop(self):
        """Main server loop running in background thread."""
        try:
            # Create and bind server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # 1 second timeout for accept()
            
            logger.info(f"📱 Waiting for cellphone connection on {self.host}:{self.port}...")
            
            while self.running:
                try:
                    # Accept client connection
                    self.client_socket, addr = self.server_socket.accept()
                    logger.info(f"✅ Cellphone connected from {addr}")
                    
                    # Handle this client
                    self._handle_client()
                    
                    logger.info(f"📱 Client disconnected, waiting for new connection...")
                    
                except socket.timeout:
                    # No connection yet, continue waiting
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                        time.sleep(1.0)
                        
        except Exception as e:
            logger.error(f"Fatal error in server loop: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
                
    def _handle_client(self):
        """Handle incoming frames from connected client."""
        try:
            self.client_socket.settimeout(5.0)  # 5 second timeout for recv
            
            while self.running:
                # Read frame size (4 bytes, network byte order = big-endian)
                size_bytes = self._recv_exact(4)
                if not size_bytes:
                    break
                    
                frame_size = struct.unpack('!I', size_bytes)[0]  # '!' = network byte order
                
                # Validate frame size (max 10MB for safety)
                if frame_size > 10 * 1024 * 1024:
                    logger.error(f"Frame too large: {frame_size} bytes, disconnecting")
                    break
                    
                # Read frame data
                frame_data = self._recv_exact(frame_size)
                if not frame_data:
                    break
                    
                # Decode JPEG to numpy array
                frame = self._decode_jpeg(frame_data)
                if frame is None:
                    logger.warning("Failed to decode frame, skipping")
                    continue
                    
                # Add to queue (drop old frames if full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Queue full, drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                        
                # Update statistics
                self.frames_received += 1
                self.bytes_received += frame_size
                
                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    delta = current_time - self.last_frame_time
                    if delta > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta)  # Exponential moving average
                self.last_frame_time = current_time
                
        except socket.timeout:
            logger.warning("Client timeout - no data received")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
                
    def _recv_exact(self, size: int) -> Optional[bytes]:
        """
        Receive exact number of bytes from socket.
        
        Args:
            size: Number of bytes to receive
            
        Returns:
            Bytes received or None if connection closed
        """
        data = b''
        while len(data) < size:
            try:
                chunk = self.client_socket.recv(size - len(data))
                if not chunk:
                    return None  # Connection closed
                data += chunk
            except socket.error as e:
                logger.error(f"Socket error while receiving: {e}")
                return None
        return data
        
    def _decode_jpeg(self, data: bytes) -> Optional[np.ndarray]:
        """
        Decode JPEG data to numpy array (BGR format).
        
        Args:
            data: JPEG encoded bytes
            
        Returns:
            Decoded frame or None if failed
        """
        try:
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Error decoding JPEG: {e}")
            return None
            
    def read_frame(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Read next frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for frame (seconds)
            
        Returns:
            Frame as numpy array (BGR) or None if timeout/no frame
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
            
    def is_connected(self) -> bool:
        """Check if a client is currently connected."""
        return self.client_socket is not None
        
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            'connected': self.is_connected(),
            'frames_received': self.frames_received,
            'bytes_received': self.bytes_received,
            'fps': self.fps,
            'queue_size': self.frame_queue.qsize(),
            'host': self.host,
            'port': self.port
        }
