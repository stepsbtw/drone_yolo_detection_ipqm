"""
Broker module for handling frame encoding and streaming.
"""

import queue
import cv2
import base64
import logging
from src.config import Config
from injector import inject

logger = logging.getLogger(__name__)


class Broker(queue.Queue):
    @inject
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.jpeg_quality = config.stream.jpeg_quality
        
    def write(self, frame):

        if frame is not None and frame.size > 0:
            # Clear queue if it has old frames (keep only latest)
            if not self.empty():
                while not self.empty():
                    try:
                        self.get_nowait()
                    except queue.Empty:
                        break
            
            try:
                # Encode frame as JPEG
                success, encoded_frame = cv2.imencode(
                    '.jpg', 
                    frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                
                if success:
                    # Convert to base64 and format as SSE (Server-Sent Events)
                    base64_frame = base64.b64encode(encoded_frame).decode('utf-8')
                    sse_data = f'data: {base64_frame}\n\n'
                    self.put_nowait(sse_data)
                else:
                    logger.warning("Failed to encode frame")
                    
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")

    def __iter__(self):
        return iter(self.get, None)
