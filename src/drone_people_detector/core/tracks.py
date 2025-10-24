"""
tracking de pessoas com deteccao de armas e filtro de kalman
"""

import uuid
from datetime import datetime
from drone_people_detector.core.estimation_submodule import Kinematic


class WeaponTrack:
    """track de arma com filtro kalman"""
    
    def __init__(self, weapon_bbox, confidence, weapon_class="weapon", track_id=None):
        if track_id is None:
            self.id = 'W-' + str(uuid.uuid4())[:8]
        else:
            self.id = track_id
        
        self.lost = False
        self.frames_alive = 0
        self.frames_since_update = 0
        self.timestamp = datetime.now()
        self.weapon_class = weapon_class
        
        # filtros de kalman para bbox da arma
        # measurement_noise baixo = confia mais nas medicoes (deteccoes)
        # process_noise baixo = assume movimento lento/suave
        # Armas: seguimos mais as deteccoes, menos as predicoes
        self.bbox_xy = Kinematic(
            measurement_noise=1,  # confia muito nas deteccoes
            process_noise=0.05,   # assume pouco movimento
            initial_gain=(1000, 1000, 1000, 1000)
        )
        self.bbox_wh = Kinematic(
            measurement_noise=1,  # confia muito nas deteccoes
            process_noise=0.05,   # assume tamanho estavel
            initial_gain=(1000, 1000, 1000, 1000)
        )
        
        self.bbox = None
        self.bbox_raw = None
        self.confidence = confidence
        self.confidence_history = [confidence]
        
        # inicializa com primeira deteccao
        if len(weapon_bbox) == 4:
            # weapon_bbox vem em formato xyxy [x1, y1, x2, y2] do detector
            # precisamos converter para xywh para o kalman filter
            x1, y1, x2, y2 = weapon_bbox
            
            # calcula xywh
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            self.bbox_xy.update(x, y)
            self.bbox_wh.update(w, h)
            
            smooth_x, smooth_y = self.bbox_xy.position
            smooth_w, smooth_h = self.bbox_wh.position
            self.bbox = [smooth_x, smooth_y, smooth_w, smooth_h]
            self.bbox_raw = weapon_bbox
    
    def update(self, detected_bbox, confidence):
        """atualiza track com nova deteccao"""
        self.lost = False
        self.frames_since_update = 0
        self.frames_alive += 1
        self.timestamp = datetime.now()
        
        self.bbox_raw = detected_bbox
        self.confidence = confidence
        self.confidence_history.append(confidence)
        
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)
        
        if len(detected_bbox) == 4:
            # weapon_bbox vem em formato xyxy [x1, y1, x2, y2] do detector
            # precisamos converter para xywh para o kalman filter
            x1, y1, x2, y2 = detected_bbox
            
            # calcula xywh
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
        
        # atualiza filtros de kalman
        self.bbox_xy.update(x, y)
        self.bbox_wh.update(w, h)
        
        smooth_x, smooth_y = self.bbox_xy.position
        smooth_w, smooth_h = self.bbox_wh.position
        self.bbox = [smooth_x, smooth_y, smooth_w, smooth_h]
    
    def predict(self):
        """chamado quando track nao e detectado no frame atual"""
        self.frames_since_update += 1
        
        # armas sao perdidas mais rapido que pessoas (5 frames)
        # balance entre evitar falsos positivos e manter continuidade
        if self.frames_since_update > 5:
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
    
    def calculate_iou(self, detected_bbox):
        """calcula iou entre track e deteccao (detected_bbox em formato xyxy)"""
        if self.bbox is None:
            return 0.0
        
        # self.bbox esta em formato xywh
        x1_min, y1_min, w1, h1 = self.bbox
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        
        # detected_bbox vem em formato xyxy [x1, y1, x2, y2]
        if len(detected_bbox) == 4:
            x2_min, y2_min, x2_max, y2_max = detected_bbox
        else:
            return 0.0
        
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
    
    def is_lost(self):
        """verifica se track deve ser removido"""
        return self.lost or self.frames_since_update > 5
    
    def get_avg_confidence(self):
        """retorna confianca media ao longo do historico"""
        if not self.confidence_history:
            return self.confidence
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def __str__(self):
        info = f"\nWeapon Track ID: {self.id}"
        info += f"\nClass: {self.weapon_class}"
        info += f"\nConfidence: {self.confidence:.2f} (avg: {self.get_avg_confidence():.2f})"
        info += f"\nFrames alive: {self.frames_alive}"
        info += f"\nFrames since update: {self.frames_since_update}"
        if self.bbox:
            info += f"\nBBox: {[int(x) for x in self.bbox]}"
        info += f"\nLost: {self.lost}"
        return info


class WeaponClassification:
    """classificacao de arma com votacao temporal"""
    
    def __init__(self, use_temporal_voting=True, confidence_threshold=0.2):
        self.timestamp = datetime.now()
        self.use_temporal_voting = use_temporal_voting
        self.confidence_threshold = confidence_threshold  # threshold para considerar armado
        self.categories = {
            'unarmed': {'votes': 0, 'confidence': 0.0},
            'armed': {'votes': 0, 'confidence': 0.0}
        }
        self.elected = None
        # para modo sem votacao temporal
        self.current_confidence = 0.0
        self.current_category = 'unarmed'
    
    def update(self, confidence, category_index):
        """atualiza classificacao: 0=desarmado, 1=armado"""
        category = 'armed' if category_index == 1 else 'unarmed'
        
        if self.use_temporal_voting:
            # modo original com votacao temporal
            self.categories[category]['votes'] += 1
            
            if confidence > self.categories[category]['confidence']:
                self.categories[category]['confidence'] = confidence
            
            armed_votes = self.categories['armed']['votes']
            unarmed_votes = self.categories['unarmed']['votes']
            
            if armed_votes > unarmed_votes:
                self.elected = ('armed', armed_votes, self.categories['armed']['confidence'])
            else:
                self.elected = ('unarmed', unarmed_votes, self.categories['unarmed']['confidence'])
        else:
            # modo sem votacao temporal - usa confianca do frame atual
            self.current_category = category
            self.current_confidence = confidence
            self.elected = (category, 1, confidence)
        
        self.timestamp = datetime.now()
    
    def has_weapon(self):
        if self.elected is None:
            return False
        category, votes, confidence = self.elected
        # usa o threshold configuravel (default 0.2)
        return category == 'armed' and confidence > self.confidence_threshold
    
    def to_string(self):
        if self.elected is None:
            return "UNKNOWN"
        category, votes, confidence = self.elected
        if self.use_temporal_voting:
            return f"{category.upper()} - {round(confidence*100, 2)}% ({votes} votes)"
        else:
            return f"{category.upper()} - {round(confidence*100, 2)}%"


class PersonTrack:
    """track de pessoa com filtro kalman e deteccao de arma"""
    
    def __init__(self, source="drone", track_id=None, use_temporal_voting=True, weapon_confidence_threshold=0.2):
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
        
        self.weapon_classifier = WeaponClassification(
            use_temporal_voting=use_temporal_voting,
            confidence_threshold=weapon_confidence_threshold
        )
        self.weapon_bboxes = []
        
        # gerenciador de tracks de armas associadas a esta pessoa
        self.weapon_tracks = []
        # IoU threshold mais alto para armas = mais rigoroso, evita falsos positivos
        self.weapon_track_manager = WeaponTrackManager(iou_threshold=0.4)
        
        self.distance = None
        self.distance_history = []
        
        # posicao geografica
        self.lat = None
        self.lon = None
        self.bearing = None
        self.x_utm = None
        self.y_utm = None
        
        self.velocity_x = 0.0
        self.velocity_y = 0.0
    
    def update(self, detected_bbox, weapon_detected=False, weapon_confidence=0.0, distance=None, weapon_bboxes=None, 
               position_data=None):
        """
        atualiza track com nova deteccao
        
        Args:
            detected_bbox: [x, y, w, h] bounding box
            weapon_detected: bool indicating weapon presence
            weapon_confidence: confidence score for weapon
            distance: estimated distance in meters
            weapon_bboxes: list of weapon bounding boxes [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': str}, ...]
            position_data: dict with 'x_utm', 'y_utm', 'lat', 'lon', 'bearing' (optional)
        """
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
        
        # atualiza tracking de armas com filtro de kalman
        if weapon_bboxes is not None and len(weapon_bboxes) > 0:
            # prepara deteccoes de arma para o track manager
            weapon_detections = []
            weapon_confidences = []
            weapon_classes = []
            
            for weapon_info in weapon_bboxes:
                if isinstance(weapon_info, dict):
                    bbox = weapon_info.get('bbox', weapon_info.get('bbox_crop', []))
                    conf = weapon_info.get('confidence', 0.0)
                    cls = weapon_info.get('class', 'weapon')
                else:
                    # fallback para formato de lista simples
                    bbox = weapon_info
                    conf = weapon_confidence
                    cls = 'weapon'
                
                weapon_detections.append(bbox)
                weapon_confidences.append(conf)
                weapon_classes.append(cls)
            
            # atualiza tracks de armas com kalman filter
            self.weapon_tracks = self.weapon_track_manager.update(
                weapon_detections, 
                weapon_confidences,
                weapon_classes
            )
            
            # mantem tambem a lista de bboxes brutos para compatibilidade
            self.weapon_bboxes = weapon_bboxes
        else:
            # sem deteccoes de arma, propaga predicoes
            self.weapon_track_manager.predict_all()
            self.weapon_tracks = self.weapon_track_manager.get_active_tracks()
            self.weapon_bboxes = []
        
        # atualiza distancia
        if distance is not None:
            self.distance = distance
            self.distance_history.append(distance)
            if len(self.distance_history) > 30:
                self.distance_history.pop(0)
        
        # atualiza posicao geografica se disponivel
        if position_data is not None:
            self.lat = position_data.get('lat')
            self.lon = position_data.get('lon')
            self.bearing = position_data.get('bearing')
            self.x_utm = position_data.get('x_utm')
            self.y_utm = position_data.get('y_utm')
    
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
    
    def weapon_track_lost(self):
        """Check if weapon classification exists but no active weapon detections in current frame"""
        has_classification = self.weapon_classifier.has_weapon()
        
        # Verifica se tem tracks que receberam deteccao NESTE frame
        # frames_since_update == 0 significa deteccao ativa
        # frames_since_update > 0 significa que esta propagando predicoes (sem nova deteccao)
        has_active_detection = any(
            wt.frames_since_update == 0 
            for wt in self.weapon_tracks
        )
        
        result = has_classification and not has_active_detection
        
        # Se tem classificacao de arma mas nenhuma deteccao ativa, marca como "lost"
        return result
    
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
    
    def __init__(self, iou_threshold=0.3, use_temporal_voting=True, weapon_confidence_threshold=0.2):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.use_temporal_voting = use_temporal_voting
        self.weapon_confidence_threshold = weapon_confidence_threshold
    
    def update(self, detections, weapon_detections=None, distances=None, weapon_bboxes_list=None, position_data_list=None):
        """
        atualiza tracks com novas deteccoes
        
        Args:
            detections: list of [x, y, w, h] bounding boxes
            weapon_detections: list of (has_weapon, confidence) tuples
            distances: list of distances in meters
            weapon_bboxes_list: list of weapon bounding boxes for each person
            position_data_list: list of position data dicts (x_utm, y_utm, lat, lon, bearing)
        """
        if weapon_detections is None:
            weapon_detections = [(False, 0.0)] * len(detections)
        
        if distances is None:
            distances = [None] * len(detections)
        
        if weapon_bboxes_list is None:
            weapon_bboxes_list = [[] for _ in detections]
        
        if position_data_list is None:
            position_data_list = [None] * len(detections)
        
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
            position_data = position_data_list[i]
            
            if best_track:
                best_track.update(detection, has_weapon, weapon_conf, distance, weapon_bboxes, position_data)
                matched_tracks.add(best_track.id)
            else:
                # cria novo track
                track_id = f"P{self.next_id:03d}"
                self.next_id += 1
                new_track = PersonTrack(
                    source="drone", 
                    track_id=track_id, 
                    use_temporal_voting=self.use_temporal_voting,
                    weapon_confidence_threshold=self.weapon_confidence_threshold
                )
                new_track.update(detection, has_weapon, weapon_conf, distance, weapon_bboxes, position_data)
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


class WeaponTrackManager:
    """gerencia multiplos tracks de armas"""
    
    def __init__(self, iou_threshold=0.25):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, weapon_detections, confidences, weapon_classes):
        """
        atualiza tracks de armas com novas deteccoes
        
        Args:
            weapon_detections: list of weapon bounding boxes [[x1,y1,x2,y2], ...]
            confidences: list of confidence scores
            weapon_classes: list of weapon class names
        """
        matched_tracks = set()
        
        # associa deteccoes aos tracks existentes
        for i, detection in enumerate(weapon_detections):
            best_track = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = track.calculate_iou(detection)
                if iou > best_iou:
                    best_track = track
                    best_iou = iou
            
            confidence = confidences[i] if i < len(confidences) else 0.0
            weapon_class = weapon_classes[i] if i < len(weapon_classes) else "weapon"
            
            if best_track:
                best_track.update(detection, confidence)
                matched_tracks.add(best_track.id)
            else:
                # cria novo track de arma
                track_id = f"W{self.next_id:03d}"
                self.next_id += 1
                new_track = WeaponTrack(detection, confidence, weapon_class, track_id=track_id)
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
    
    def predict_all(self):
        """propaga predicao para todos os tracks quando nao ha deteccoes"""
        for track in self.tracks.values():
            track.predict()
        
        # remove tracks perdidos
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if not track.is_lost()
        }
    
    def get_active_tracks(self):
        """retorna lista de tracks de armas ativos"""
        return [t for t in self.tracks.values() if not t.lost]
    
    def get_all_tracks(self):
        """retorna todos os tracks de armas incluindo os recentemente perdidos"""
        return list(self.tracks.values())
    
    def reset(self):
        """reseta todos os tracks de armas"""
        self.tracks = {}
        self.next_id = 0
