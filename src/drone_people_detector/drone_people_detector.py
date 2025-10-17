"""
detector principal para identificacao de pessoas e armas em videos de drone
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

from drone_people_detector.core.enhanced_detector import EnhancedPeopleDetector
from drone_people_detector.core.estimation import Camera

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
    
    def draw_bbox(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """desenha bounding boxes e labels no frame"""
        
    def get_camera(self) -> Camera:
        """
        Get the camera object for distance estimation.
        
        Returns:
            Camera instance
        """
        return self.camera
    
    def get_predict(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        Get the next frame and predictions.
        Similar to ShipDetectorClassifier.get_predict()
        
        Returns:
            Tuple of (frame, tracks_list)
            - frame: numpy array with the image, or None if no more frames
            - tracks_list: list of track dictionaries with detection info
        """
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
    
    def draw_bbox(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        Similar to ShipDetectorClassifier.draw_bbox()
        
        Args:
            frame: Input frame (numpy array)
            tracks: List of PersonTrack objects or track dictionaries
            
        Returns:
            Annotated frame with bounding boxes and labels
        """
        if frame is None or tracks is None:
            return frame
        
        annotated_frame = frame.copy()
        
        for track in tracks:
            # extrai info do track - suporta PersonTrack objects e dicts
            if hasattr(track, 'bbox') and not hasattr(track, 'get'):
                # objeto PersonTrack
                bbox_xywh = track.bbox
                if bbox_xywh is None:
                    continue
                x, y, w, h = bbox_xywh
                bbox = [x, y, x + w, y + h]
                track_id = track.id
                confidence = 0.0
                has_weapon = track.weapon_classifier.has_weapon() if hasattr(track, 'weapon_classifier') else False
                weapon_conf = track.weapon_classifier.categories['armed']['confidence'] if has_weapon else 0.0
                distance = track.distance if hasattr(track, 'distance') else None
                weapon_bboxes = track.weapon_bboxes if hasattr(track, 'weapon_bboxes') else []
            else:
                # formato dicionario
                bbox = track.get('bbox')
                track_id = track.get('track_id', 'Unknown')
                confidence = track.get('confidence', 0.0)
                has_weapon = track.get('has_weapon', False)
                weapon_conf = track.get('weapon_confidence', 0.0)
                distance = track.get('distance')
                weapon_bboxes = track.get('weapon_bboxes', [])
            
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # bbox da pessoa (verde, linha grossa)
            person_color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), person_color, 3)
            
            # bbox da arma se disponivel (vermelho, linha fina)
            if weapon_bboxes:
                for weapon_bbox in weapon_bboxes:
                    wx1, wy1, wx2, wy2 = map(int, weapon_bbox)
                    cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                    
                    weapon_label = f"ARMA {int(weapon_conf*100)}%"
                    (wlabel_w, wlabel_h), _ = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, 
                                (wx1, wy1 - wlabel_h - 8), 
                                (wx1 + wlabel_w + 8, wy1),
                                (0, 0, 200), -1)
                    cv2.putText(annotated_frame, weapon_label, (wx1 + 4, wy1 - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # label da pessoa - organizado em linhas
            label_lines = []
            label_lines.append(f"ID: {track_id}")
            
            if distance is not None:
                label_lines.append(f"Dist: {distance:.1f}m")
            
            if has_weapon and not weapon_bboxes:
                label_lines.append(f"ARMADO ({int(weapon_conf*100)}%)")
            
            # calcula dimensoes do label
            line_height = 20
            max_width = 0
            for line in label_lines:
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_width = max(max_width, w)
            
            total_height = len(label_lines) * line_height + 10
            label_bg_color = (0, 180, 0)
            
            # desenha fundo do label
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - total_height),
                (x1 + max_width + 15, y1),
                label_bg_color,
                -1
            )
            
            # Draw each line
            # desenha cada linha do label
            y_offset = y1 - total_height + 18
            for line in label_lines:
                cv2.putText(
                    annotated_frame,
                    line,
                    (x1 + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                y_offset += line_height
        
        return annotated_frame
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
    
    def __del__(self):
        self.release()
