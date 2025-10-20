from injector import Module, Binder, singleton

from src.config import Config
from src.services.broker.broker import Broker
from src.services.streamer import VideoStreamer

class AppModule(Module):
    def configure(self, binder: Binder):
        # Configuração
        binder.bind(Config, to=Config, scope=singleton)
        
        # Serviços core
        binder.bind(Broker, to=Broker, scope=singleton)
        binder.bind(VideoStreamer, to=VideoStreamer, scope=singleton)