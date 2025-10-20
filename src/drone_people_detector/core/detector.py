import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    from drone_people_detector.core.tracks import PersonTrack, TrackManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .tracks import PersonTrack, TrackManager

from drone_people_detector.core.camera import Camera
from drone_people_detector.core.weapon_detector import WeaponDetector
from drone_people_detector.core.monocular_vision_submodule import MonocularVision


class DetectionStatistics:
    def __init__(self):
        self.total_people_detected = 0
        self.total_weapons_detected = 0
        self.frames_with_people = 0
        self.frames_with_weapons = 0
        self.unique_people_tracked = 0


class Detector:
    """detector de pessoas com tracking e deteccao de armas"""
    
    def __init__(self, model_path, confidence_threshold=0.5, 
                 enable_weapon_detection=True, weapon_confidence_threshold=0.2,
                 weapon_model_path='models/weapons/yolov8guns.pt',
                 sample_majority_threshold=1, enable_tracking=True,
                 show_weapon_bbox=False):
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        self.enable_weapon_detection = enable_weapon_detection
        self.weapon_confidence_threshold = weapon_confidence_threshold
        self.sample_majority_threshold = sample_majority_threshold
        self.show_weapon_bbox = show_weapon_bbox
        
        if self.enable_weapon_detection:
            self.weapon_detector = WeaponDetector(
                model_path=weapon_model_path,
                confidence_threshold=weapon_confidence_threshold
            )
        
        self.enable_tracking = enable_tracking
        self.track_manager = TrackManager(iou_threshold=0.3)
        
        self.statistics = DetectionStatistics()
        
        # Track video-level processing
        self.current_video_tracks = []
        self.frame_count = 0
    
    def process_frame_with_tracking(self, frame, camera=None):
        """processa frame com tracking de pessoas e armas"""
        self.frame_count += 1
        
        # detecta pessoas
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        weapon_detections = []
        weapon_bboxes = []
        distances = []
        
        position_data_list = []
        
        for result in results:
            for detection in result.boxes:
                bbox = detection.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                person_bbox = [x1, y1, w, h]
                detections.append(person_bbox)
                
                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                has_weapon = False
                weapon_conf = 0.0
                weapon_bbox_list = []
                
                if self.enable_weapon_detection and person_crop.size > 0:
                    offset = (int(x1), int(y1)) if self.show_weapon_bbox else None
                    _, detections_info, _ = self.weapon_detector.detect_weapons(person_crop, offset)
                    has_weapon = len(detections_info) > 0
                    if has_weapon:
                        weapon_conf = detections_info[0]['confidence']
                        if self.show_weapon_bbox:
                            weapon_bbox_list = [det['bbox'] for det in detections_info]
                
                weapon_detections.append((has_weapon, weapon_conf))
                weapon_bboxes.append(weapon_bbox_list)
                
                distance = None
                position_data = None
                if camera and h > 0:
                    # Estimate distance using monocular vision method
                    # Assuming average person height of 1.7m
                    AVERAGE_PERSON_HEIGHT_M = 1.7
                    detected_bbox = [x1, y1, w, h]  # [x, y, width, height]
                    
                    # Use monocular_vision_detection_method_2 for complete position estimation
                    # Returns: (x_utm, y_utm, lat, lon, bearing, distance)
                    x_utm, y_utm, lat, lon, bearing, distance = MonocularVision.monocular_vision_detection_method_2(
                        camera, AVERAGE_PERSON_HEIGHT_M, detected_bbox
                    )
                    
                    # Store position data
                    position_data = {
                        'x_utm': x_utm,
                        'y_utm': y_utm,
                        'lat': lat,
                        'lon': lon,
                        'bearing': bearing
                    }
                
                distances.append(distance)
                position_data_list.append(position_data)
        
        if self.enable_tracking:
            tracks = self.track_manager.update(
                detections, 
                weapon_detections, 
                distances,
                weapon_bboxes if self.show_weapon_bbox else None,
                position_data_list
            )
        else:
            tracks = []
            for i, det in enumerate(detections):
                track = PersonTrack(track_id=f"T{i}")
                has_weapon, weapon_conf = weapon_detections[i]
                w_bboxes = weapon_bboxes[i] if self.show_weapon_bbox else []
                pos_data = position_data_list[i] if i < len(position_data_list) else None
                track.update(det, has_weapon, weapon_conf, distances[i], w_bboxes, pos_data)
                tracks.append(track)
        
        annotated_frame = self._annotate_frame(frame.copy(), tracks)
        return annotated_frame, tracks
    
    def _annotate_frame(self, frame, tracks):
        """desenha annotations no frame"""
        for track in tracks:
            if track.lost:
                continue
            
            bbox = track.get_bbox('xyxy')
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # bbox da pessoa (verde, linha grossa)
            person_color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 3)
            
            # bbox da arma se disponivel (vermelho, linha fina)
            if self.show_weapon_bbox and hasattr(track, 'weapon_bboxes') and track.weapon_bboxes:
                weapon_conf = track.weapon_classifier.categories['armed']['confidence'] if track.has_weapon() else 0.0
                for weapon_bbox in track.weapon_bboxes:
                    wx1, wy1, wx2, wy2 = map(int, weapon_bbox)
                    cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                    
                    weapon_label = f"ARMA {int(weapon_conf*100)}%"
                    (wlabel_w, wlabel_h), _ = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, 
                                (wx1, wy1 - wlabel_h - 8), 
                                (wx1 + wlabel_w + 8, wy1),
                                (0, 0, 200), -1)
                    cv2.putText(frame, weapon_label, (wx1 + 4, wy1 - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # label da pessoa - organizado em linhas
            label_lines = []
            label_lines.append(f"ID: {track.id}")
            
            if track.distance:
                label_lines.append(f"Dist: {track.distance:.1f}m")
            
            # adiciona bearing se disponivel
            if track.bearing is not None:
                label_lines.append(f"Bear: {track.bearing:.0f}°")
            
            # adiciona coordenadas geograficas se disponiveis
            if track.lat is not None and track.lon is not None:
                label_lines.append(f"Lat: {track.lat:.6f}")
                label_lines.append(f"Lon: {track.lon:.6f}")
            
            # mostra status de arma apenas se nao houver bbox separada
            if track.has_weapon() and not (self.show_weapon_bbox and track.weapon_bboxes):
                weapon_conf = track.weapon_classifier.categories['armed']['confidence']
                label_lines.append(f"ARMADO ({int(weapon_conf*100)}%)")
            
            # calcula dimensoes
            line_height = 20
            max_width = 0
            for line in label_lines:
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_width = max(max_width, w)
            
            total_height = len(label_lines) * line_height + 10
            label_bg_color = (0, 180, 0)
            
            # desenha fundo do label
            cv2.rectangle(
                frame, 
                (x1, y1 - total_height), 
                (x1 + max_width + 15, y1), 
                label_bg_color, 
                -1
            )
            
            # desenha cada linha do label
            y_offset = y1 - total_height + 18
            for line in label_lines:
                cv2.putText(
                    frame, line, (x1 + 8, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )
                y_offset += line_height
            
            # desenha seta de velocidade se em movimento
            speed = (track.velocity_x**2 + track.velocity_y**2)**0.5
            if speed > 5:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                arrow_end_x = int(center_x + track.velocity_x * 0.1)
                arrow_end_y = int(center_y + track.velocity_y * 0.1)
                cv2.arrowedLine(
                    frame, 
                    (center_x, center_y), 
                    (arrow_end_x, arrow_end_y),
                    (255, 255, 0), 2, tipLength=0.3
                )
        
        # info do frame
        info_text = f"Frame: {self.frame_count} | Tracks: {len([t for t in tracks if not t.lost])}"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return frame
    
    def process_video_with_tracking(self, video_path, output_path=None, camera=None):
        """processa video completo com tracking"""
        print(f"Processing video with tracking: {video_path}")
        
        # reseta tracking para novo video
        self.track_manager.reset()
        self.frame_count = 0
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        stats = {
            'total_frames': 0,
            'frames_with_people': 0,
            'unique_tracks': set(),
            'frames_with_weapons': 0,
            'max_people_in_frame': 0,
            'track_durations': {}
        }
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame, tracks = self.process_frame_with_tracking(frame, camera)
                
                stats['total_frames'] += 1
                if tracks:
                    stats['frames_with_people'] += 1
                    stats['max_people_in_frame'] = max(
                        stats['max_people_in_frame'], 
                        len([t for t in tracks if not t.lost])
                    )
                    
                    for track in tracks:
                        stats['unique_tracks'].add(track.id)
                        if track.has_weapon():
                            stats['frames_with_weapons'] += 1
                            break
                        
                        if track.id not in stats['track_durations']:
                            stats['track_durations'][track.id] = 0
                        stats['track_durations'][track.id] += 1
                
                if out:
                    out.write(annotated_frame)
                
                if stats['total_frames'] % 30 == 0:
                    print(f"  Processed {stats['total_frames']}/{total_frames} frames...", end='\r')
        
        finally:
            cap.release()
            if out:
                out.release()
        
        print(f"\nCompleted! Processed {stats['total_frames']} frames")
        
        stats['unique_people_count'] = len(stats['unique_tracks'])
        stats['avg_track_duration'] = (
            sum(stats['track_durations'].values()) / len(stats['track_durations'])
            if stats['track_durations'] else 0
        )
        
        return stats
    
    def reset_tracking(self):
        """reseta estado do tracking"""
        self.track_manager.reset()
        self.frame_count = 0
