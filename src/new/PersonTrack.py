"""
tracking de pessoas com deteccao de armas e filtro de kalman
"""

import uuid
from datetime import datetime
from new.estimation_submodule import Kinematic


class WeaponClassification:
    """classificacao de arma com votacao temporal"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.categories = {
            'unarmed': {'votes': 0, 'confidence': 0.0},
            'armed': {'votes': 0, 'confidence': 0.0}
        }
        self.elected = None
    
    def update(self, confidence, category_index):
        """atualiza classificacao: 0=desarmado, 1=armado"""
        category = 'armed' if category_index == 1 else 'unarmed'
        
        self.categories[category]['votes'] += 1
        
        if confidence > self.categories[category]['confidence']:
            self.categories[category]['confidence'] = confidence
        
        armed_votes = self.categories['armed']['votes']
        unarmed_votes = self.categories['unarmed']['votes']
        
        if armed_votes > unarmed_votes:
            self.elected = ('armed', armed_votes, self.categories['armed']['confidence'])
        else:
            self.elected = ('unarmed', unarmed_votes, self.categories['unarmed']['confidence'])
        
        self.timestamp = datetime.now()
    
    def has_weapon(self):
        if self.elected is None:
            return False
        category, votes, confidence = self.elected
        return category == 'armed' and confidence > 0.3
    
    def to_string(self):
        if self.elected is None:
            return "UNKNOWN"
        category, votes, confidence = self.elected
        return f"{category.upper()} - {round(confidence*100, 2)}% ({votes} votes)"


class PersonTrack:
    """track de pessoa com filtro kalman e deteccao de arma"""
    
    def __init__(self, source="drone", track_id=None):
        if track_id is None:
            self.id = source + '-' + str(uuid.uuid4())[:8]
        else:
            self.id = track_id
        
        self.lost = False
        self.frames_alive = 0
        self.frames_since_update = 0
        self.timestamp = datetime.now()
        
        # filtros de kalman para bbox
        self.bbox_xy = Kinematic(
            measurement_noise=5,
            process_noise=0.15,
            initial_gain=(1000, 1000, 1000, 1000)
        )
        self.bbox_wh = Kinematic(
            measurement_noise=5,
            process_noise=0.15,
            initial_gain=(1000, 1000, 1000, 1000)
        )
        
        self.bbox = None
        self.bbox_raw = None
        
        self.weapon_classifier = WeaponClassification()
        self.weapon_bboxes = []
        
        self.distance = None
        self.distance_history = []
        
        self.lat = None
        self.lon = None
        self.bearing = None
        
        self.velocity_x = 0.0
        self.velocity_y = 0.0
    
    def update(self, detected_bbox, weapon_detected=False, weapon_confidence=0.0, distance=None, weapon_bboxes=None):
        """atualiza track com nova deteccao"""
        self.lost = False
        self.frames_since_update = 0
        self.frames_alive += 1
        self.timestamp = datetime.now()
        
        self.bbox_raw = detected_bbox
        
        if len(detected_bbox) == 4:
            x, y, w, h = detected_bbox
            
            # verifica se formato esta correto (xywh vs xyxy)
            if w > 5000 or h > 5000:
                x1, y1, x2, y2 = detected_bbox
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
        
        # atualiza filtros de kalman
        self.bbox_xy.update(x, y)
        self.bbox_wh.update(w, h)
        
        smooth_x, smooth_y = self.bbox_xy.position
        smooth_w, smooth_h = self.bbox_wh.position
        self.bbox = [smooth_x, smooth_y, smooth_w, smooth_h]
        
        self.velocity_x, self.velocity_y = self.bbox_xy.velocity
        
        # atualiza classificacao de arma
        category_index = 1 if weapon_detected else 0
        self.weapon_classifier.update(weapon_confidence, category_index)
        
        # atualiza distancia
        if distance is not None:
            self.distance = distance
            self.distance_history.append(distance)
            if len(self.distance_history) > 30:
                self.distance_history.pop(0)
        
        self.weapon_bboxes = weapon_bboxes if weapon_bboxes is not None else []
    
    def predict(self):
        """chamado quando track nao e detectado no frame atual"""
        self.frames_since_update += 1
        
        if self.frames_since_update > 30:
            self.lost = True
    
    def get_bbox(self, format='xywh'):
        """retorna bbox no formato solicitado"""
        if self.bbox is None:
            return None
        
        x, y, w, h = self.bbox
        
        if format == 'xywh':
            return [x, y, w, h]
        elif format == 'xyxy':
            return [x, y, x + w, y + h]
        else:
            return self.bbox
    
    def get_smoothed_bbox(self):
        """retorna bbox filtrado por kalman"""
        return self.get_bbox('xywh')
    
    def calculate_iou(self, detected_bbox):
        """calcula iou entre track e deteccao"""
        if self.bbox is None:
            return 0.0
        
        x1_min, y1_min, w1, h1 = self.bbox
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        
        if len(detected_bbox) == 4:
            x2, y2, w2, h2 = detected_bbox
            if w2 > 5000 or h2 > 5000:
                x2_min, y2_min, x2_max, y2_max = detected_bbox
            else:
                x2_min, y2_min = x2, y2
                x2_max = x2_min + w2
                y2_max = y2_min + h2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Calculate union
        area1 = w1 * h1
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def has_weapon(self):
        """Check if person is classified as armed"""
        return self.weapon_classifier.has_weapon()
    
    def is_lost(self):
        """verifica se track deve ser removido"""
        return self.lost or self.frames_since_update > 30
    
    def get_status_string(self):
        """retorna string de status para exibicao"""
        status = f"ID:{self.id}"
        
        if self.has_weapon():
            status += " [ARMED]"
        else:
            status += " [UNARMED]"
        
        if self.distance:
            status += f" {self.distance:.1f}m"
        
        if self.bearing:
            status += f" @ {self.bearing:.0f}°"
        
        speed = (self.velocity_x**2 + self.velocity_y**2)**0.5
        if speed > 5:
            status += f" (moving)"
        
        return status
    
    def get_avg_distance(self):
        """retorna distancia media ao longo do historico"""
        if not self.distance_history:
            return self.distance
        return sum(self.distance_history) / len(self.distance_history)
    
    def __str__(self):
        info = f"\nTrack ID: {self.id}"
        info += f"\nFrames alive: {self.frames_alive}"
        info += f"\nFrames since update: {self.frames_since_update}"
        info += f"\nWeapon: {self.weapon_classifier.to_string()}"
        if self.distance:
            info += f"\nDistance: {self.distance:.2f}m"
        if self.bbox:
            info += f"\nBBox: {[int(x) for x in self.bbox]}"
        info += f"\nLost: {self.lost}"
        return info


class TrackManager:
    """gerencia multiplos tracks de pessoas"""
    
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, detections, weapon_detections=None, distances=None, weapon_bboxes_list=None):
        """atualiza tracks com novas deteccoes"""
        if weapon_detections is None:
            weapon_detections = [(False, 0.0)] * len(detections)
        
        if distances is None:
            distances = [None] * len(detections)
        
        if weapon_bboxes_list is None:
            weapon_bboxes_list = [[] for _ in detections]
        
        matched_tracks = set()
        
        # associa deteccoes aos tracks existentes
        for i, detection in enumerate(detections):
            best_track = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = track.calculate_iou(detection)
                if iou > best_iou:
                    best_track = track
                    best_iou = iou
            
            has_weapon, weapon_conf = weapon_detections[i]
            distance = distances[i]
            weapon_bboxes = weapon_bboxes_list[i]
            
            if best_track:
                best_track.update(detection, has_weapon, weapon_conf, distance, weapon_bboxes)
                matched_tracks.add(best_track.id)
            else:
                # cria novo track
                track_id = f"P{self.next_id:03d}"
                self.next_id += 1
                new_track = PersonTrack(source="drone", track_id=track_id)
                new_track.update(detection, has_weapon, weapon_conf, distance, weapon_bboxes)
                self.tracks[track_id] = new_track
        
        # marca tracks nao associados como perdidos
        for track in self.tracks.values():
            if track.id not in matched_tracks:
                track.predict()
        
        # remove tracks completamente perdidos
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if not track.is_lost()
        }
        
        return list(self.tracks.values())
    
    def get_active_tracks(self):
        """retorna lista de tracks ativos (nao perdidos)"""
        return [t for t in self.tracks.values() if not t.lost]
    
    def get_all_tracks(self):
        """retorna todos os tracks incluindo os recentemente perdidos"""
        return list(self.tracks.values())
    
    def reset(self):
        """reseta todos os tracks"""
        self.tracks = {}
        self.next_id = 0

