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
    parser.add_argument('--input_with_weapons', default='inputs_with_weapons/samples', 
                       help='Input directory containing sample folders')
    parser.add_argument('--input_without_weapons', default='inputs_without_weapons/samples', 
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
        
    if not os.path.exists(args.input_with_weapons):
        print(f"Error: Input directory not found: {args.input_with_weapons}")
        return 1
    
    if not os.path.exists(args.input_without_weapons):
        print(f"Error: Input directory not found: {args.input_without_weapons}")
        return 1
    
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    print(f"Initializing detector with model: {args.model}")
    detector = PeopleDetector(args.model, args.confidence, enable_weapon_detection=enable_weapons)
    
    # Set crop saving preference
    detector.save_crops = save_crops
    
    # Process all sample directories
    print(f"Processing samples from: {args.input_with_weapons}")
    print(f"Processing samples from: {args.input_without_weapons}")
    print(f"Output will be saved to: {args.output}")
    print(f"Crop saving: {'Enabled' if save_crops else 'Disabled'}")
    print(f"Weapon detection: {'Enabled' if enable_weapons and detector.enable_weapon_detection else 'Disabled'}")
    
    # Check if input is a single directory with images or a parent directory with subdirectories
    if os.path.isdir(args.input_with_weapons) and os.path.isdir(args.input_without_weapons):
        # Check if input directory contains image files directly
        detector.process_all_sample_directories(args.input_with_weapons, args.output, True)
        #detector.process_all_sample_directories(args.input_without_weapons, args.output, False)
    else:
        print(f"Error: Input path does not exist: {args.input_with_weapons} or {args.input_without_weapons}")
        return 1

    # Print comprehensive statistics
    detector.stats.print_summary()
    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
