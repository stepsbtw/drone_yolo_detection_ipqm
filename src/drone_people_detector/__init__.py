"""
Drone People Detector Library

A library for detecting people and weapons in drone footage using YOLO models.
Provides tracking, distance estimation, and video processing capabilities.
"""

from drone_people_detector.drone_people_detector import DronePeopleDetector
from drone_people_detector.core.estimation import Camera
from new.PersonTrack import TrackManager

__version__ = "1.0.0"

__all__ = [
    'DronePeopleDetector',
    'Camera',
    'TrackManager',
]
