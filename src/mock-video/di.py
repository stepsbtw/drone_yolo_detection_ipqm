from injector import Module, Binder, singleton

from broker import Broker

class AppModule(Module):
    def configure(self, binder: Binder):
        binder.bind(Broker, to=Broker, scope=singleton)