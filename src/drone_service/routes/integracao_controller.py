import drone_people_detector
import logging
from flask import request
from flask_restx import Resource, Namespace, abort
from werkzeug.exceptions import ClientDisconnected
import time
from injector import Injector
import logging

from di import AppModule
from src.config import Config
from src.services.streamer import VideoStreamer
from src.services.broker.broker import Broker

logger = logging.getLogger(__name__)

injector = Injector([AppModule()])
    
config = injector.get(Config)
streamer = injector.get(VideoStreamer)
streamer.start()

drone_detector = drone_people_detector.DronePeopleDetector(config)

integracao_ns = Namespace('integracao-controller', description='Integração Controller')

@integracao_ns.route('/stream')
class StreamResource(Resource):
    def post(self):
        broker: Broker = injector.get(Broker) 

        buffer = ''
        try:
            while True:
                chunk = request.input_stream.read(1024).decode('utf-8')
                if not chunk:
                    break

                buffer += chunk
                while '\n\n' in buffer:
                    segment, buffer = buffer.split('\n\n', 1)
                    
                    img_to_show, tracks_list = drone_detector.get_classification(segment)

                    annotated_frame = drone_detector.draw_bbox(img_to_show, tracks_list)
                    broker.write(annotated_frame)
                    time.sleep(0.1)

        except (ClientDisconnected, Exception) as e:
            logging.error(f"Error while processing stream: {e}")
            abort(500, "Internal Server Error")