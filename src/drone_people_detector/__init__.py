"""
Drone People Detector Library

A library for detecting people and weapons in drone footage using YOLO models.
Provides tracking, distance estimation, and video processing capabilities.
"""

from drone_people_detector.drone_people_detector import DronePeopleDetector
from drone_people_detector.core.camera import Camera
from drone_people_detector.core.tracks import TrackManager, PersonTrack, WeaponTrack, WeaponTrackManager
from drone_people_detector.core.detector import Detector
from drone_people_detector.core.weapon_detector import WeaponDetector

__version__ = "1.0.0"

__all__ = [
    'DronePeopleDetector',
    'Camera',
    'TrackManager',
    'PersonTrack',
    'WeaponTrack',
    'WeaponTrackManager',
    'Detector',
    'WeaponDetector',
]
