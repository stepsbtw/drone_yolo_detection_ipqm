"""
detector principal para identificacao de pessoas e armas em videos de drone
"""

import os
from pathlib import Path
import sys
import logging
import cv2
import numpy as np
from typing import Any, Tuple, Optional, List, Dict
from dotenv import load_dotenv

import base64
from drone_people_detector.core.detector import Detector
from drone_people_detector.core.camera import Camera
from drone_people_detector.core.viewer import _draw_bbox, add_frame_info

logger = logging.getLogger(__name__)


class DronePeopleDetector:
    """detector de pessoas e armas com tracking"""
    
    def __init__(self, config: Any):
        self.config = config
        
        # inicializa camera (Autel EVO II Dual V2)
        self.camera = Camera(
            image_width_px=getattr(config.camera, 'image_width_px', 1920),
            image_height_px=getattr(config.camera, 'image_height_px', 1080),
            sensor_width_mm=getattr(config.camera, 'sensor_width_mm', None),  # Uses default 6.17mm
            sensor_height_mm=getattr(config.camera, 'sensor_height_mm', None),  # Uses default 4.55mm
            focal_35mm_mm=getattr(config.camera, 'focal_35mm_mm', None),  # Uses default 25.6mm
            bearing=getattr(config.camera, 'bearing', 0),
            lat=getattr(config.camera, 'latitude', getattr(config.camera, 'lat', None)),
            lon=getattr(config.camera, 'longitude', getattr(config.camera, 'lon', None)),
            zoom=getattr(config.camera, 'zoom', 1.0)
        )
        
        # Initialize detector
        self.detector = Detector(
            model_path=getattr(config.detection, 'model_path_people', 'models/people/yolo11n.pt'),
            confidence_threshold=getattr(config.detection, 'person_confidence', 0.5),
            enable_tracking=getattr(config.detection, 'enable_tracking', True),
            enable_weapon_detection=getattr(config.detection, 'enable_weapon_detection', True),
            weapon_model_path=getattr(config.detection, 'model_path_weapons', 'models/weapons/yolov8guns.pt'),
            weapon_confidence_threshold=getattr(config.detection, 'weapon_confidence', 0.2),
            show_weapon_bbox=getattr(config.detection, 'show_weapon_bbox', False)
        )
        
        # Video capture (if video path provided)
        self.cap = None
        if hasattr(config, 'video') and hasattr(config.video, 'input_path') and config.video.input_path is not None:
            self.cap = cv2.VideoCapture(config.video.input_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {config.video.input_path}")
            logger.info(f"Video opened: {config.video.input_path}")
        
        self._current_frame = None
        self._current_tracks = None
    
    def get_camera(self) -> Camera:
        """retorna instancia da camera"""
        return self.camera
    
    def get_classification(self, frame: str) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        processa frame codificado em base64 e retorna frame processado + tracks
        usado para processar frames recebidos via streaming
        """
        if frame is None:
            return None, None

        try:
            # decodifica frame base64
            bytes_data = base64.b64decode(frame.replace('data: ', ''))
            np_array = np.frombuffer(bytes_data, np.uint8)
            arr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            # processa frame com tracking
            annotated_frame, tracks = self.detector.process_frame_with_tracking(arr, self.camera)
            return annotated_frame, tracks

        except Exception:
            logger.exception("Erro em get_classification")
            return None, None
    
    def get_predict(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        le proximo frame e retorna frame original + tracks
        anotacao deve ser feita separadamente com draw_bbox()
        """
        if self.cap is None:
            logger.warning("No video capture initialized")
            return None, None
        
        # le frame
        ret, frame = self.cap.read()
        if not ret:
            logger.info("End of video or failed to read frame")
            return None, None
        
        # processa frame com tracking
        annotated_frame, tracks = self.detector.process_frame_with_tracking(
            frame, 
            self.camera
        )
        
        # armazena frame original e tracks
        self._current_frame = frame.copy()
        self._current_tracks = tracks
        
        # retorna frame ORIGINAL (nao anotado) e tracks
        # anotacao sera feita separadamente em draw_bbox()
        return frame, tracks
    
    def draw_bbox(self, frame: np.ndarray, tracks: List, 
                  frame_count: int = 0, fps: float = 0.0) -> np.ndarray:
        """
        desenha bboxes profissionais com texto de alta qualidade
        estilo ship-detector-classifier
        """
        if frame is None or not tracks:
            return frame
        
        # conta pessoas e armas
        people_count = len(tracks)
        weapons_count = sum(1 for t in tracks if self._has_weapon(t))
        
        # desenha bboxes
        annotated_frame = _draw_bbox(frame, tracks)
        
        # adiciona info do frame
        annotated_frame = add_frame_info(
            annotated_frame, 
            frame_count, 
            people_count, 
            weapons_count, 
            fps
        )
        
        return annotated_frame
    
    def _has_weapon(self, track):
        """verifica se track tem arma detectada"""
        if hasattr(track, 'weapon_classifier'):
            return track.weapon_classifier.has_weapon()
        elif isinstance(track, dict):
            return track.get('has_weapon', False)
        return False
    
    def release(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
    
    def __del__(self):
        self.release()
