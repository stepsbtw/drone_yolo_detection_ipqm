#!/usr/bin/env python3
"""
Main script to test distance estimation and RMSE metrics using the drone_people_detector library.
"""

import argparse
import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import math

# Use the installed library
from drone_people_detector.core.camera import Camera
from ultralytics import YOLO


class DistanceEstimationTester:
    """Test distance estimation."""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize with YOLO model and camera."""
        # Load detection model
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        self.camera = Camera()  # Default Autel EVO II camera
        
        # Statistics
        self.distance_pairs = []  # (estimated, real) tuples
        self.distance_pairs_by_height = {}  # {camera_height: [(est, real), ...]}
        self.distance_pairs_by_distance = {}  # {real_distance: [(est, real), ...]}
        
    def extract_metadata_from_filename(self, filepath):
        """Extract real distance, camera height, and class from filename/directory.
        
        Pattern: class_distance_height_clip_number
        Example: falso_05_02_clip_000 -> class='falso', distance=5m, height=2m
        """
        dir_path = os.path.dirname(filepath)
        dir_name = os.path.basename(dir_path)
        
        if not dir_name:
            dir_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Extract class (real or falso)
        sample_class = 'real' if dir_name.lower().startswith('real') else 'falso'
        
        # Extract numbers from directory name
        import re
        numbers = re.findall(r'\d{2,3}', dir_name)
        
        real_distance = None
        camera_height = None
        
        if len(numbers) >= 2:
            try:
                real_distance = float(numbers[0])
                camera_height = float(numbers[1])
            except ValueError:
                pass
        
        return sample_class, real_distance, camera_height
    
    def compute_rmse(self, pairs):
        """Compute RMSE from (estimated, real) pairs."""
        if not pairs:
            return None
        
        squared_errors = [(est - real) ** 2 for est, real in pairs if real is not None]
        if not squared_errors:
            return None
        
        mse = sum(squared_errors) / len(squared_errors)
        return math.sqrt(mse)
    
    def process_image(self, image_path):
        """Process a single image and estimate distances."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        # Extract metadata
        sample_class, real_distance, camera_height = self.extract_metadata_from_filename(image_path)
        
        # Run YOLO detection
        results = self.model(image, conf=self.confidence_threshold, classes=[0], verbose=False)
        
        people_detected = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence >= self.confidence_threshold:
                        people_detected += 1
                        
                        # Calculate pixel height
                        pixel_height = y2 - y1
                        
                        # Use simple distance estimation
                        distance_est = self.camera.estimate_distance(pixel_height, real_height_m=1.7)
                        
                        # Store pairs for RMSE calculation
                        if real_distance is not None:
                            pair = (distance_est, real_distance)
                            self.distance_pairs.append(pair)
                            
                            # Store by camera height
                            if camera_height is not None:
                                if camera_height not in self.distance_pairs_by_height:
                                    self.distance_pairs_by_height[camera_height] = []
                                self.distance_pairs_by_height[camera_height].append(pair)
                            
                            # Store by real distance
                            if real_distance not in self.distance_pairs_by_distance:
                                self.distance_pairs_by_distance[real_distance] = []
                            self.distance_pairs_by_distance[real_distance].append(pair)
                        
                        # Calculate error
                        error = abs(distance_est - real_distance) if real_distance else 0
                        
                        # Print detection info
                        print(f"  Person {people_detected}: Estimated={distance_est:.2f}m, Real={real_distance}m, "
                              f"Error={error:.2f}m, CamHeight={camera_height}m")
        
        if people_detected == 0:
            print(f"  No people detected")
        
        return people_detected
    
    def process_directory(self, input_dir):
        """Process all images in a directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        
        print(f"\nProcessing directory: {input_dir}")
        print(f"Found {len(image_files)} files")
        
        for image_path in sorted(image_files):
            print(f"\n{os.path.basename(image_path)}:")
            self.process_image(image_path)
    
    def process_all_samples(self, samples_dir):
        """Process all sample directories."""
        sample_dirs = [d for d in os.listdir(samples_dir)
                      if os.path.isdir(os.path.join(samples_dir, d))]
        
        print(f"Found {len(sample_dirs)} sample directories")
        
        for sample_dir in sorted(sample_dirs):
            input_path = os.path.join(samples_dir, sample_dir)
            self.process_directory(input_path)
    
    def print_summary(self):
        """Print comprehensive RMSE summary."""
        print("\n" + "=" * 80)
        print("DISTANCE ESTIMATION AND RMSE SUMMARY")
        print("=" * 80)
        
        # Overall RMSE
        overall_rmse = self.compute_rmse(self.distance_pairs)
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total measurements: {len(self.distance_pairs)}")
        if overall_rmse is not None:
            print(f"  Overall RMSE: {overall_rmse:.3f} meters")
        else:
            print(f"  Overall RMSE: N/A")
        
        # RMSE by camera height
        if self.distance_pairs_by_height:
            print(f"\nRMSE BY CAMERA HEIGHT:")
            for height in sorted(self.distance_pairs_by_height.keys()):
                pairs = self.distance_pairs_by_height[height]
                rmse = self.compute_rmse(pairs)
                print(f"  Height {height}m: RMSE={rmse:.3f}m (n={len(pairs)})")
        
        # RMSE by real distance
        if self.distance_pairs_by_distance:
            print(f"\nRMSE BY REAL DISTANCE:")
            for distance in sorted(self.distance_pairs_by_distance.keys()):
                pairs = self.distance_pairs_by_distance[distance]
                rmse = self.compute_rmse(pairs)
                print(f"  Distance {distance}m: RMSE={rmse:.3f}m (n={len(pairs)})")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test distance estimation'
    )
    parser.add_argument('--model', default='models/people/yolo11n.pt',
                       help='Path to YOLO model file for people detection')
    parser.add_argument('--input', default='inputs/samples',
                       help='Input directory containing sample folders')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return 1
    
    # Initialize tester
    print("=" * 80)
    print("DISTANCE ESTIMATION TEST")
    print("=" * 80)
    print(f"Detection model: {args.model}")
    print(f"Camera: Autel EVO II Dual V2 (default settings)")
    print(f"Confidence threshold: {args.confidence}")
    print("")
    
    tester = DistanceEstimationTester(args.model, args.confidence)
    
    # Check if input is a single directory or parent with subdirectories
    if os.path.isdir(args.input):
        subdirs = [d for d in os.listdir(args.input)
                  if os.path.isdir(os.path.join(args.input, d))]
        
        if subdirs:
            # Process all sample directories
            tester.process_all_samples(args.input)
        else:
            # Process single directory
            tester.process_directory(args.input)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    # Print comprehensive RMSE summary
    tester.print_summary()
    print("\n✅ Processing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
