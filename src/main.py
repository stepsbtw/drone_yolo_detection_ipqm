#!/usr/bin/env python3
"""
Main script to run people detection pipeline.
"""

import argparse
import os
import sys
import glob
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import PeopleDetector


def main():
    parser = argparse.ArgumentParser(description='Detect people in images using YOLOv11')
    parser.add_argument('--model', default='models/yolo11n.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--input', default='inputs/samples', 
                       help='Input directory containing sample folders')
    parser.add_argument('--output', default='output/detections', 
                       help='Output directory for processed images')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--save-crops', action='store_true', default=True,
                       help='Save individual person crops (default: True)')
    parser.add_argument('--no-crops', action='store_true',
                       help='Disable saving individual person crops')
    parser.add_argument('--no-weapons', action='store_true',
                       help='Disable weapon detection in person crops')
    
    args = parser.parse_args()
    
    # Handle crop saving logic
    save_crops = args.save_crops and not args.no_crops
    
    # Handle weapon detection logic
    enable_weapons = not args.no_weapons
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
        
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return 1
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"Initializing detector with model: {args.model}")
    detector = PeopleDetector(args.model, args.confidence, enable_weapon_detection=enable_weapons)
    
    # Set crop saving preference
    detector.save_crops = save_crops
    
    # Process all sample directories
    print(f"Processing samples from: {args.input}")
    print(f"Output will be saved to: {args.output}")
    print(f"Crop saving: {'Enabled' if save_crops else 'Disabled'}")
    print(f"Weapon detection: {'Enabled' if enable_weapons and detector.enable_weapon_detection else 'Disabled'}")
    
    # Check if input is a single directory with images or a parent directory with subdirectories
    if os.path.isdir(args.input):
        # Check if input directory contains image files directly
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
        
        if image_files:
            # Input directory contains images directly - process as single directory
            print(f"Processing single directory with {len(image_files)} images")
            detector.process_directory(args.input, args.output)
        else:
            # Input directory contains subdirectories - process all subdirectories
            detector.process_all_sample_directories(args.input, args.output)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())