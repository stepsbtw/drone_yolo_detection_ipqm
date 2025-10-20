"""
Video Streamer module for sending frames to REST API.
"""

import time
import threading
import requests
import logging
from typing import Optional

from src.services.broker.broker import Broker
from src.config import Config
from injector import inject

logger = logging.getLogger(__name__)


class VideoStreamer:

    @inject
    def __init__(self, config: Config, broker: Broker):
        self.broker = broker
        self.config = config
        self.stream_connected: Optional[bool] = None
        self._stop_flag = False
        
    def start(self):
        threading.Thread(target=self._stream_loop, daemon=True).start()
        logger.info("Video streamer started")
        
    def stop(self):
        self._stop_flag = True
        logger.info("Video streamer stopped")
        
    def _stream_loop(self):
        while not self._stop_flag and self.stream_connected is None:
            try:
                logger.info(f"Attempting to connect to: {self.config.stream.url}")
                
                # Prepare headers
                headers = {
                    'Content-Type': 'text/event-stream',
                    'Authorization': f'Apikey {self.config.stream.apikey}',
                    'Camera-Id': self.config.camera.id,
                    'Camera-Name': self.config.camera.name,
                    'Camera-Latitude': str(self.config.camera.latitude),
                    'Camera-Longitude': str(self.config.camera.longitude)
                }
                
                # Send streaming request
                response = requests.post(
                    self.config.stream.url,
                    data=self.broker,
                    headers=headers,
                    stream=True,
                    verify=False,
                    timeout=30
                )
                
                if response.status_code == 200:
                    self.stream_connected = True
                    logger.info(f"✓ Stream connected successfully")
                else:
                    logger.warning(f"Stream connection failed with status: {response.status_code}")
                    time.sleep(5)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"[Streamer] Connection error: {e}")
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"[Streamer] Unexpected error: {e}")
                time.sleep(5)
                
    def is_connected(self) -> bool:
        return self.stream_connected is True
