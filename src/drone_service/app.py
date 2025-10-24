from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api

from routes.integracao_controller import integracao_ns
from config import Config

app = Flask(__name__)
CORS(app, origins=['*'], methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])

api = Api(
    app,
    version='1.0',
    title='SERVIÇO DE CLASSIFICAÇÃO DE ARMAS',
    description='Serviço responsável por classificar pessoas armadas.',
    doc='/api-docs'
)

api.add_namespace(integracao_ns, path=f'{Config.BASE_PATH}/v1/integracao')

if __name__ == '__main__':
    app.run()