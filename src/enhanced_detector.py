"""
Enhanced Detector with Multi-Object Tracking
Wraps the existing PeopleDetector with PersonTrack for persistent tracking
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from detector import PeopleDetector, DetectionStatistics
from new.PersonTrack import PersonTrack, TrackManager
from estimation import Camera


class EnhancedPeopleDetector(PeopleDetector):
    """
    Enhanced detector that adds multi-object tracking with Kalman filtering
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, 
                 enable_weapon_detection=True, weapon_confidence_threshold=0.2,
                 sample_majority_threshold=1, enable_tracking=True):
        """
        Initialize enhanced detector
        
        Args:
            model_path: Path to person detection model
            confidence_threshold: Confidence threshold for person detection
            enable_weapon_detection: Enable weapon detection
            weapon_confidence_threshold: Confidence threshold for weapon detection
            sample_majority_threshold: Frames needed for weapon classification
            enable_tracking: Enable multi-object tracking (recommended)
        """
        # Initialize parent detector
        super().__init__(
            model_path, 
            confidence_threshold,
            enable_weapon_detection,
            weapon_confidence_threshold,
            sample_majority_threshold
        )
        
        # Tracking components
        self.enable_tracking = enable_tracking
        self.track_manager = TrackManager(iou_threshold=0.3)
        
        # Track video-level processing
        self.current_video_tracks = []
        self.frame_count = 0
    
    def process_frame_with_tracking(self, frame, camera=None):
        """
        Process a single frame with tracking
        
        Args:
            frame: Input frame (numpy array)
            camera: Optional Camera object for distance estimation
        
        Returns:
            annotated_frame: Frame with tracking annotations
            tracks: List of active PersonTrack objects
        """
        self.frame_count += 1
        
        # Detect people
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        weapon_detections = []
        distances = []
        
        for result in results:
            for detection in result.boxes:
                # Get person bbox
                bbox = detection.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                # Convert to xywh format for tracking
                person_bbox = [x1, y1, w, h]
                detections.append(person_bbox)
                
                # Crop person
                person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Weapon detection
                has_weapon = False
                weapon_conf = 0.0
                
                if self.enable_weapon_detection and person_crop.size > 0:
                    weapon_results = self.weapon_detector.detect(person_crop)
                    has_weapon = len(weapon_results) > 0
                    if has_weapon:
                        weapon_conf = weapon_results[0].conf.item()
                
                weapon_detections.append((has_weapon, weapon_conf))
                
                # Distance estimation
                distance = None
                if camera and h > 0:
                    distance = camera.estimate_distance(h)
                distances.append(distance)
        
        # Update tracks
        if self.enable_tracking:
            tracks = self.track_manager.update(
                detections, 
                weapon_detections, 
                distances
            )
        else:
            # No tracking - create temporary tracks
            tracks = []
            for i, det in enumerate(detections):
                track = PersonTrack(track_id=f"T{i}")
                has_weapon, weapon_conf = weapon_detections[i]
                track.update(det, has_weapon, weapon_conf, distances[i])
                tracks.append(track)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame.copy(), tracks)
        
        return annotated_frame, tracks
    
    def _annotate_frame(self, frame, tracks):
        """
        Annotate frame with track information
        
        Args:
            frame: Input frame
            tracks: List of PersonTrack objects
        
        Returns:
            Annotated frame
        """
        for track in tracks:
            if track.lost:
                continue
            
            bbox = track.get_bbox('xyxy')
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on weapon status
            if track.has_weapon():
                color = (0, 0, 255)  # Red for armed
                label_bg_color = (0, 0, 200)
            else:
                color = (0, 255, 0)  # Green for unarmed
                label_bg_color = (0, 200, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = track.get_status_string()
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w + 10, y1), 
                label_bg_color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # Draw velocity indicator if moving
            speed = (track.velocity_x**2 + track.velocity_y**2)**0.5
            if speed > 5:  # threshold in pixels/sec
                # Draw arrow
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                arrow_end_x = int(center_x + track.velocity_x * 0.1)
                arrow_end_y = int(center_y + track.velocity_y * 0.1)
                cv2.arrowedLine(
                    frame, 
                    (center_x, center_y), 
                    (arrow_end_x, arrow_end_y),
                    color, 2, tipLength=0.3
                )
        
        # Add frame info
        info_text = f"Frame: {self.frame_count} | Tracks: {len([t for t in tracks if not t.lost])}"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return frame
    
    def process_video_with_tracking(self, video_path, output_path=None, camera=None):
        """
        Process entire video with tracking
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            camera: Optional Camera object for distance estimation
        
        Returns:
            Dictionary with statistics and track information
        """
        print(f"Processing video with tracking: {video_path}")
        
        # Reset tracking for new video
        self.track_manager.reset()
        self.frame_count = 0
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video writer
        out = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Statistics
        stats = {
            'total_frames': 0,
            'frames_with_people': 0,
            'unique_tracks': set(),
            'frames_with_weapons': 0,
            'max_people_in_frame': 0,
            'track_durations': {}
        }
        
        # Process video
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, tracks = self.process_frame_with_tracking(frame, camera)
                
                # Update statistics
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
                            break  # Count frame once
                        
                        # Track duration
                        if track.id not in stats['track_durations']:
                            stats['track_durations'][track.id] = 0
                        stats['track_durations'][track.id] += 1
                
                # Write output
                if out:
                    out.write(annotated_frame)
                
                # Progress
                if stats['total_frames'] % 30 == 0:
                    print(f"  Processed {stats['total_frames']}/{total_frames} frames...", end='\r')
        
        finally:
            cap.release()
            if out:
                out.release()
        
        print(f"\nCompleted! Processed {stats['total_frames']} frames")
        
        # Finalize statistics
        stats['unique_people_count'] = len(stats['unique_tracks'])
        stats['avg_track_duration'] = (
            sum(stats['track_durations'].values()) / len(stats['track_durations'])
            if stats['track_durations'] else 0
        )
        
        return stats
    
    def reset_tracking(self):
        """Reset tracking state (call between videos)"""
        self.track_manager.reset()
        self.frame_count = 0


def demo_tracking(video_path, output_path):
    """
    Demo function showing tracking capabilities
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
    """
    print("=== Enhanced Detector with Tracking Demo ===\n")
    
    # Initialize detector
    detector = EnhancedPeopleDetector(
        model_path='../models/people/yolo11n.pt',
        confidence_threshold=0.5,
        enable_weapon_detection=True,
        weapon_confidence_threshold=0.2,
        enable_tracking=True
    )
    
    # Initialize camera for distance estimation
    camera = Camera(
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
        focal_35mm_mm=25.6,
        image_width_px=1920,
        image_height_px=1080
    )
    
    # Process video
    stats = detector.process_video_with_tracking(
        video_path, 
        output_path, 
        camera
    )
    
    # Print results
    if stats:
        print("\n=== Video Statistics ===")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with people: {stats['frames_with_people']}")
        print(f"Unique people tracked: {stats['unique_people_count']}")
        print(f"Max people in frame: {stats['max_people_in_frame']}")
        print(f"Frames with weapons: {stats['frames_with_weapons']}")
        print(f"Average track duration: {stats['avg_track_duration']:.1f} frames")
        
        print("\n=== Individual Tracks ===")
        for track_id, duration in sorted(stats['track_durations'].items()):
            print(f"  {track_id}: {duration} frames ({duration/stats['total_frames']*100:.1f}%)")


if __name__ == '__main__':
    # Demo usage
    import sys
    
    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        demo_tracking(video_path, output_path)
    else:
        print("Usage: python enhanced_detector.py <input_video> <output_video>")
        print("\nExample:")
        print("  python enhanced_detector.py inputs/clips/test.mp4 output/tracked_test.mp4")
