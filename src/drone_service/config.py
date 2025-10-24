import os
import logging

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    BASE_PATH = '/api'
    
LOGGING = {
    'pika': logging.DEBUG,
    'console': logging.DEBUG,
    'logger': logging.DEBUG
}

REST = {
    'ptz_url': os.getenv('PTZ_URL', 'http://10.7.40.191:33024/api/v1/integracao/ptz'),
    'stream_url': os.getenv('STREAM_URL', 'http://10.7.40.191:33024/api/v1/integracao/stream'),
    'calibration_url': os.getenv('CALIBRATION_URL', 'http://localhost:33023/api/v1/integracao/calibration'),
    'apikey': os.getenv('APIKEY', 'aGlkcmEtYXBpa2V5LjE0OTgucHFJMlpUS3ZjS29YNnZJM3VZZnBUaHhDWTZET2JVOWNBZ0F4UkhIbFhHTkFIYmZqaFhp')
}