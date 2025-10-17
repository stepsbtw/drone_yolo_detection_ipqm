"""
Enhanced Person Track for Drone Weapon Detection
Combines multi-object tracking with Kalman filtering and weapon classification
"""

import uuid
from datetime import datetime
from new.estimation_submodule import Kinematic


class WeaponClassification:
    """Simplified weapon classification with voting"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.categories = {
            'unarmed': {'votes': 0, 'confidence': 0.0},
            'armed': {'votes': 0, 'confidence': 0.0}
        }
        self.elected = None
    
    def update(self, confidence, category_index):
        """
        Update weapon classification
        Args:
            confidence: Detection confidence (0-1)
            category_index: 0=unarmed, 1=armed
        """
        category = 'armed' if category_index == 1 else 'unarmed'
        
        # Increment votes
        self.categories[category]['votes'] += 1
        
        # Update confidence (keep max)
        if confidence > self.categories[category]['confidence']:
            self.categories[category]['confidence'] = confidence
        
        # Elect category with most votes
        armed_votes = self.categories['armed']['votes']
        unarmed_votes = self.categories['unarmed']['votes']
        
        if armed_votes > unarmed_votes:
            self.elected = ('armed', armed_votes, self.categories['armed']['confidence'])
        else:
            self.elected = ('unarmed', unarmed_votes, self.categories['unarmed']['confidence'])
        
        self.timestamp = datetime.now()
    
    def has_weapon(self):
        """Returns True if person is classified as armed"""
        if self.elected is None:
            return False
        category, votes, confidence = self.elected
        return category == 'armed' and confidence > 0.3
    
    def to_string(self):
        """Get classification string"""
        if self.elected is None:
            return "UNKNOWN"
        category, votes, confidence = self.elected
        return f"{category.upper()} - {round(confidence*100, 2)}% ({votes} votes)"


class PersonTrack:
    """Enhanced person track with Kalman filtering and weapon detection"""
    
    def __init__(self, source="drone", track_id=None):
        """
        Initialize a person track
        Args:
            source: Source identifier (e.g., "drone")
            track_id: Optional custom track ID
        """
        if track_id is None:
            self.id = source + '-' + str(uuid.uuid4())[:8]
        else:
            self.id = track_id
        
        # State tracking
        self.lost = False
        self.frames_alive = 0
        self.frames_since_update = 0
        self.timestamp = datetime.now()
        
        # Bounding box tracking with Kalman filters
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
        
        # Current bbox (smoothed)
        self.bbox = None
        self.bbox_raw = None  # Raw detection bbox
        
        # Weapon classification with temporal voting
        self.weapon_classifier = WeaponClassification()
        
        # Distance estimation
        self.distance = None
        self.distance_history = []
        
        # Optional GPS tracking (if drone has GPS)
        self.lat = None
        self.lon = None
        self.bearing = None
        
        # Velocity tracking
        self.velocity_x = 0.0
        self.velocity_y = 0.0
    
    def update(self, detected_bbox, weapon_detected=False, weapon_confidence=0.0, distance=None):
        """
        Update track with new detection
        
        Args:
            detected_bbox: [x, y, w, h] or [x1, y1, x2, y2]
            weapon_detected: Boolean indicating if weapon was detected
            weapon_confidence: Confidence of weapon detection
            distance: Optional estimated distance in meters
        """
        self.lost = False
        self.frames_since_update = 0
        self.frames_alive += 1
        self.timestamp = datetime.now()
        
        # Store raw bbox
        self.bbox_raw = detected_bbox
        
        # Convert to [x, y, w, h] format if needed
        if len(detected_bbox) == 4:
            x, y, w, h = detected_bbox
            # Check if it's [x1, y1, x2, y2] format (w and h would be large)
            if w > 500 or h > 500:  # Likely x2, y2 format
                x1, y1, x2, y2 = detected_bbox
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
        
        # Update Kalman filters
        self.bbox_xy.update(x, y)
        self.bbox_wh.update(w, h)
        
        # Get smoothed bbox
        smooth_x, smooth_y = self.bbox_xy.position
        smooth_w, smooth_h = self.bbox_wh.position
        self.bbox = [smooth_x, smooth_y, smooth_w, smooth_h]
        
        # Get velocity (pixels per second)
        self.velocity_x, self.velocity_y = self.bbox_xy.velocity
        
        # Update weapon classification
        category_index = 1 if weapon_detected else 0
        self.weapon_classifier.update(weapon_confidence, category_index)
        
        # Update distance
        if distance is not None:
            self.distance = distance
            self.distance_history.append(distance)
            # Keep only last 30 distances
            if len(self.distance_history) > 30:
                self.distance_history.pop(0)
    
    def predict(self):
        """Called when track is not detected in current frame"""
        self.frames_since_update += 1
        # Kalman filter will predict position automatically on next update
        
        if self.frames_since_update > 30:  # Lost after 30 frames
            self.lost = True
    
    def get_bbox(self, format='xywh'):
        """
        Get bounding box
        Args:
            format: 'xywh' or 'xyxy'
        Returns:
            bbox in requested format
        """
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
        """Get Kalman-filtered bounding box"""
        return self.get_bbox('xywh')
    
    def calculate_iou(self, detected_bbox):
        """
        Calculate IoU between track and detection
        Args:
            detected_bbox: [x, y, w, h] or [x1, y1, x2, y2]
        Returns:
            IoU value (0-1)
        """
        if self.bbox is None:
            return 0.0
        
        # Convert both to xyxy format
        x1_min, y1_min, w1, h1 = self.bbox
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        
        # Handle detected_bbox format
        if len(detected_bbox) == 4:
            x2, y2, w2, h2 = detected_bbox
            if w2 > 500 or h2 > 500:  # Likely xyxy format
                x2_min, y2_min, x2_max, y2_max = detected_bbox
            else:  # xywh format
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
        """Check if track should be removed"""
        return self.lost or self.frames_since_update > 30
    
    def get_status_string(self):
        """Get status string for display"""
        status = f"ID:{self.id}"
        
        if self.has_weapon():
            status += " [ARMED]"
        else:
            status += " [UNARMED]"
        
        if self.distance:
            status += f" {self.distance:.1f}m"
        
        if self.bearing:
            status += f" @ {self.bearing:.0f}°"
        
        # Add velocity if moving
        speed = (self.velocity_x**2 + self.velocity_y**2)**0.5
        if speed > 5:  # pixels/sec threshold
            status += f" (moving)"
        
        return status
    
    def get_avg_distance(self):
        """Get average distance over history"""
        if not self.distance_history:
            return self.distance
        return sum(self.distance_history) / len(self.distance_history)
    
    def __str__(self):
        """String representation"""
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
    """Manages multiple person tracks"""
    
    def __init__(self, iou_threshold=0.3):
        """
        Initialize track manager
        Args:
            iou_threshold: Minimum IoU for track-detection matching
        """
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
    
    def update(self, detections, weapon_detections=None, distances=None):
        """
        Update tracks with new detections
        
        Args:
            detections: List of bboxes [[x, y, w, h], ...]
            weapon_detections: List of (has_weapon, confidence) tuples
            distances: List of estimated distances in meters
        
        Returns:
            List of active tracks
        """
        if weapon_detections is None:
            weapon_detections = [(False, 0.0)] * len(detections)
        
        if distances is None:
            distances = [None] * len(detections)
        
        matched_tracks = set()
        
        # Match detections to existing tracks
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
            
            # Update existing track or create new one
            has_weapon, weapon_conf = weapon_detections[i]
            distance = distances[i]
            
            if best_track:
                best_track.update(detection, has_weapon, weapon_conf, distance)
                matched_tracks.add(best_track.id)
            else:
                # Create new track
                track_id = f"P{self.next_id:03d}"
                self.next_id += 1
                new_track = PersonTrack(source="drone", track_id=track_id)
                new_track.update(detection, has_weapon, weapon_conf, distance)
                self.tracks[track_id] = new_track
        
        # Mark unmatched tracks as lost
        for track in self.tracks.values():
            if track.id not in matched_tracks:
                track.predict()
        
        # Remove completely lost tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if not track.is_lost()
        }
        
        return list(self.tracks.values())
    
    def get_active_tracks(self):
        """Get list of active (not lost) tracks"""
        return [t for t in self.tracks.values() if not t.lost]
    
    def get_all_tracks(self):
        """Get all tracks including recently lost ones"""
        return list(self.tracks.values())
    
    def reset(self):
        """Reset all tracks"""
        self.tracks = {}
        self.next_id = 0
