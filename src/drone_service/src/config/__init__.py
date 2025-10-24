"""
Configuration module for drone service.
"""

import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


class CameraConfig:
    """Camera/Drone configuration."""
    def __init__(self):
        self.id = os.getenv('CAMERA_ID', 'drone-01')
        self.name = os.getenv('CAMERA_NAME', 'Drone Camera 01')
        self.latitude = float(os.getenv('CAMERA_LATITUDE', '-23.550520'))
        self.longitude = float(os.getenv('CAMERA_LONGITUDE', '-46.633308'))
        
        # Camera parameters for distance estimation
        self.sensor_width_mm = float(os.getenv('SENSOR_WIDTH_MM', '6.4'))
        self.sensor_height_mm = float(os.getenv('SENSOR_HEIGHT_MM', '4.8'))
        self.focal_35mm_mm = float(os.getenv('FOCAL_35MM_MM', '25.6'))
        self.image_width_px = int(os.getenv('IMAGE_WIDTH_PX', '1920'))
        self.image_height_px = int(os.getenv('IMAGE_HEIGHT_PX', '1080'))


class VideoConfig:
    """Video input configuration."""
    def __init__(self):
        self.input_path = os.getenv('VIDEO_INPUT', 'inputs/clips/video.mp4')
        self.loop = os.getenv('VIDEO_LOOP', 'true').lower() == 'true'
        self.fps_limit = int(os.getenv('VIDEO_FPS_LIMIT', '10'))
        
        # Socket mode configuration
        self.use_socket = os.getenv('USE_SOCKET', 'false').lower() == 'true'
        self.socket_port = int(os.getenv('SOCKET_PORT', '5555'))


class DetectionConfig:
    """Detection configuration."""
    def __init__(self):
        self.enable_tracking = os.getenv('ENABLE_TRACKING', 'true').lower() == 'true'
        self.enable_weapon_detection = os.getenv('ENABLE_WEAPON_DETECTION', 'true').lower() == 'true'
        self.show_weapon_bbox = os.getenv('SHOW_WEAPON_BBOX', 'false').lower() == 'true'
        self.use_temporal_voting = os.getenv('USE_TEMPORAL_VOTING', 'true').lower() == 'true'
        self.person_confidence = float(os.getenv('PERSON_CONFIDENCE', '0.5'))
        self.weapon_confidence = float(os.getenv('WEAPON_CONFIDENCE', '0.2'))
        self.model_path_people = os.getenv('MODEL_PEOPLE', 'models/people/yolo11n.pt')
        self.model_path_weapons = os.getenv('MODEL_WEAPONS', 'models/weapons/yolov8guns.pt')


class StreamConfig:
    """Streaming configuration."""
    def __init__(self):
        self.url = os.getenv('STREAM_URL')
        self.apikey = os.getenv('APIKEY')
        self.jpeg_quality = int(os.getenv('JPEG_QUALITY', '80'))


class Config:
    """Main configuration class."""
    def __init__(self):
        self.camera = CameraConfig()
        self.video = VideoConfig()
        self.detection = DetectionConfig()
        self.stream = StreamConfig()
        self.frame_timeout = int(os.getenv('FRAME_TIMEOUT', '10'))
