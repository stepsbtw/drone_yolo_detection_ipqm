import time
import threading
import requests
import logging
from injector import inject

from broker import Broker

class VideoStreamer:
    @inject
    def __init__(self, broker: Broker):
        self.broker = broker
        self.stream_connected = None

    def start(self):
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stream_loop(self):
        while self.stream_connected is None:
            try:
                response = requests.post(
                    'http://localhost:5000/api/v1/integracao/stream',
                    data=self.broker,
                    headers={
                        'Content-Type': 'text/event-stream',
                        'Camera-Id': 'Drone',
                        'Camera-Name': 'Drone'
                    },
                    stream=True,
                    verify=False
                )
                
                if response.status_code == 200:
                    self.stream_connected = True
                    
            except Exception as e:
                logging.error(f"[Streamer] Falha ao enviar: {e}")
                time.sleep(1)
