"""
Socket Video Capture - cv2.VideoCapture-like interface for socket streams.
Makes socket video receiver compatible with existing drone_people_detector code.
"""

import numpy as np
from typing import Tuple, Optional
import logging

from .socket_video_receiver import SocketVideoReceiver

logger = logging.getLogger(__name__)


class SocketVideoCapture:
    """
    cv2.VideoCapture-compatible wrapper for socket video receiver.
    
    This allows the drone_people_detector library to work with socket streams
    without any modifications to the library code.
    
    Usage:
        cap = SocketVideoCapture(port=5555)
        ret, frame = cap.read()  # Just like cv2.VideoCapture
        cap.release()
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5555):
        """
        Initialize socket video capture.
        
        Args:
            host: Host address to bind
            port: Port to listen on
        """
        self.receiver = SocketVideoReceiver(host=host, port=port)
        self.receiver.start()
        self._is_opened = True
        
        logger.info(f"SocketVideoCapture initialized on port {port}")
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from socket stream.
        
        Compatible with cv2.VideoCapture.read()
        
        Returns:
            Tuple (ret, frame) where:
            - ret: True if frame was successfully read
            - frame: Frame as numpy array (BGR) or None
        """
        if not self._is_opened:
            return False, None
            
        frame = self.receiver.read_frame(timeout=5.0)
        if frame is not None:
            return True, frame
        else:
            # No frame received (timeout or disconnected)
            return False, None
            
    def isOpened(self) -> bool:
        """Check if capture is opened."""
        return self._is_opened
        
    def release(self):
        """Release the socket receiver."""
        if self._is_opened:
            self.receiver.stop()
            self._is_opened = False
            logger.info("SocketVideoCapture released")
            
    def get(self, propId: int) -> float:
        """
        Get property value (for compatibility with cv2.VideoCapture).
        
        Limited implementation - returns sensible defaults.
        """
        # CAP_PROP_FRAME_WIDTH = 3
        if propId == 3:
            return 1920.0
        # CAP_PROP_FRAME_HEIGHT = 4
        elif propId == 4:
            return 1080.0
        # CAP_PROP_FPS = 5
        elif propId == 5:
            return self.receiver.fps
        else:
            return 0.0
            
    def set(self, propId: int, value: float) -> bool:
        """
        Set property value (for compatibility).
        
        Not implemented for socket streams.
        """
        logger.warning(f"set() not implemented for SocketVideoCapture")
        return False
        
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return self.receiver.get_stats()
        
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.receiver.is_connected()
        
    def __del__(self):
        """Cleanup on deletion."""
        self.release()
