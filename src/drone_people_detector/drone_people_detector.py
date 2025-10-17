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

# Adiciona o diretório do módulo ao sys.path
module_path = Path(__file__).resolve().parent
sys.path.insert(0, str(module_path))

from core.enhanced_detector import EnhancedPeopleDetector
from core.estimation import Camera
from core.viewer import _draw_bbox, add_frame_info

logger = logging.getLogger(__name__)


class DronePeopleDetector:
    """detector de pessoas e armas com tracking"""
    
    def __init__(self, config: Any):
        self.config = config
        
        # inicializa camera
        self.camera = Camera(
            sensor_width_mm=getattr(config.camera, 'sensor_width_mm', 6.4),
            sensor_height_mm=getattr(config.camera, 'sensor_height_mm', 4.8),
            focal_35mm_mm=getattr(config.camera, 'focal_35mm_mm', 25.6),
            image_width_px=getattr(config.camera, 'image_width_px', 1920),
            image_height_px=getattr(config.camera, 'image_height_px', 1080)
        )
        
        # Initialize detector
        self.detector = EnhancedPeopleDetector(
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
        if hasattr(config, 'video') and hasattr(config.video, 'input_path'):
            self.cap = cv2.VideoCapture(config.video.input_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {config.video.input_path}")
            logger.info(f"Video opened: {config.video.input_path}")
        
        self._current_frame = None
        self._current_tracks = None
        
    def get_camera(self) -> Camera:
        return self.camera
    
    def get_predict(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """retorna proximo frame e predicoes"""
        if self.cap is None:
            logger.warning("No video capture initialized")
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.info("End of video or failed to read frame")
            return None, None
        
        # processa frame com tracking
        annotated_frame, tracks = self.detector.process_frame_with_tracking(
            frame, 
            self.camera
        )
        
        self._current_frame = frame.copy()
        self._current_tracks = tracks
        
        
        return frame, tracks
    
    def get_camera(self) -> Camera:
        return self.camera
    
    def get_predict(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        if self.cap is None:
            logger.warning("No video capture initialized")
            return None, None
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            logger.info("End of video or failed to read frame")
            return None, None
        
        # Process frame
        annotated_frame, tracks = self.detector.process_frame_with_tracking(
            frame, 
            self.camera
        )
        
        # Store for later use
        self._current_frame = frame.copy()  # Original frame
        self._current_tracks = tracks
        
        # Return original frame (not annotated) and tracks
        # Annotation is done separately in draw_bbox()
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
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
    
    def __del__(self):
        self.release()
