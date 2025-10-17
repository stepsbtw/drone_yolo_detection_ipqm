#!/usr/bin/env python3
"""
Test script for enhanced tracking system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_detector import EnhancedPeopleDetector
from estimation import Camera


def test_tracking_on_clip(clip_path, output_dir='output/tracked_clips'):
    """
    Test tracking on a single clip
    
    Args:
        clip_path: Path to input video clip
        output_dir: Directory to save output
    """
    print(f"\n{'='*70}")
    print(f"Testing tracking on: {Path(clip_path).name}")
    print(f"{'='*70}\n")
    
    # Setup output path
    output_path = Path(output_dir) / f"tracked_{Path(clip_path).name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector with tracking
    detector = EnhancedPeopleDetector(
        model_path='models/people/yolo11n.pt',
        confidence_threshold=0.5,
        enable_weapon_detection=True,
        weapon_confidence_threshold=0.2,
        sample_majority_threshold=1,
        enable_tracking=True  # Enable tracking!
    )
    
    # Initialize camera for distance estimation
    camera = Camera(
        sensor_width_mm=6.4,
        sensor_height_mm=4.8,
        focal_35mm_mm=25.6,
        image_width_px=1920,
        image_height_px=1080
    )
    
    # Process video with tracking
    stats = detector.process_video_with_tracking(
        clip_path,
        output_path,
        camera
    )
    
    # Print results
    if stats:
        print(f"\n{'='*70}")
        print("TRACKING RESULTS")
        print(f"{'='*70}")
        print(f"Total frames processed:    {stats['total_frames']}")
        print(f"Frames with people:        {stats['frames_with_people']} ({stats['frames_with_people']/stats['total_frames']*100:.1f}%)")
        print(f"Unique people tracked:     {stats['unique_people_count']}")
        print(f"Max people in frame:       {stats['max_people_in_frame']}")
        print(f"Frames with weapons:       {stats['frames_with_weapons']}")
        print(f"Average track duration:    {stats['avg_track_duration']:.1f} frames")
        
        if stats['track_durations']:
            print(f"\nIndividual Track Durations:")
            print(f"{'-'*50}")
            for track_id in sorted(stats['track_durations'].keys()):
                duration = stats['track_durations'][track_id]
                percentage = (duration / stats['total_frames']) * 100
                print(f"  {track_id:8s}: {duration:4d} frames ({percentage:5.1f}%)")
        
        print(f"\n✓ Output saved to: {output_path}")
    
    return stats


def test_multiple_clips(num_clips=3):
    """Test tracking on multiple clips"""
    
    clips_dir = Path('inputs/clips')
    
    if not clips_dir.exists():
        print(f"Error: Clips directory not found: {clips_dir}")
        return
    
    # Get all clips
    clips = sorted(list(clips_dir.glob('*.mp4')))[:num_clips]
    
    if not clips:
        print(f"Error: No clips found in {clips_dir}")
        return
    
    print(f"\n{'#'*70}")
    print(f"TESTING ENHANCED TRACKING ON {len(clips)} CLIPS")
    print(f"{'#'*70}")
    
    all_stats = []
    
    for clip in clips:
        stats = test_tracking_on_clip(clip)
        if stats:
            all_stats.append((clip.name, stats))
    
    # Print summary
    print(f"\n{'#'*70}")
    print("SUMMARY ACROSS ALL CLIPS")
    print(f"{'#'*70}\n")
    
    total_people = sum(s['unique_people_count'] for _, s in all_stats)
    total_frames = sum(s['total_frames'] for _, s in all_stats)
    total_with_people = sum(s['frames_with_people'] for _, s in all_stats)
    
    print(f"Clips processed:           {len(all_stats)}")
    print(f"Total frames:              {total_frames}")
    print(f"Frames with people:        {total_with_people} ({total_with_people/total_frames*100:.1f}%)")
    print(f"Total unique people:       {total_people}")
    print(f"Avg people per clip:       {total_people/len(all_stats):.1f}")
    
    print(f"\n{'='*70}")
    print("Detailed Breakdown:")
    print(f"{'='*70}")
    for clip_name, stats in all_stats:
        print(f"\n{clip_name}:")
        print(f"  People: {stats['unique_people_count']:2d} | "
              f"Frames: {stats['total_frames']:3d} | "
              f"Max in frame: {stats['max_people_in_frame']:2d}")


def compare_tracking_vs_no_tracking(clip_path):
    """
    Compare results with and without tracking
    """
    print(f"\n{'#'*70}")
    print("COMPARING: TRACKING vs NO TRACKING")
    print(f"{'#'*70}\n")
    
    from estimation import Camera
    
    camera = Camera()
    
    # Test WITH tracking
    print("\n--- WITH TRACKING ---")
    detector_with = EnhancedPeopleDetector(
        model_path='models/people/yolo11n.pt',
        confidence_threshold=0.5,
        enable_weapon_detection=True,
        enable_tracking=True
    )
    stats_with = detector_with.process_video_with_tracking(
        clip_path,
        'output/comparison/with_tracking.mp4',
        camera
    )
    
    # Test WITHOUT tracking
    print("\n--- WITHOUT TRACKING ---")
    detector_without = EnhancedPeopleDetector(
        model_path='models/people/yolo11n.pt',
        confidence_threshold=0.5,
        enable_weapon_detection=True,
        enable_tracking=False
    )
    stats_without = detector_without.process_video_with_tracking(
        clip_path,
        'output/comparison/without_tracking.mp4',
        camera
    )
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<30} | {'With Tracking':>15} | {'Without Tracking':>15}")
    print(f"{'-'*70}")
    print(f"{'Unique people detected':<30} | {stats_with['unique_people_count']:>15d} | {stats_without['unique_people_count']:>15d}")
    print(f"{'Avg track duration (frames)':<30} | {stats_with['avg_track_duration']:>15.1f} | {stats_without['avg_track_duration']:>15.1f}")
    print(f"{'Max people in frame':<30} | {stats_with['max_people_in_frame']:>15d} | {stats_without['max_people_in_frame']:>15d}")
    
    print(f"\n✓ Videos saved:")
    print(f"  - With tracking:    output/comparison/with_tracking.mp4")
    print(f"  - Without tracking: output/comparison/without_tracking.mp4")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced tracking system')
    parser.add_argument('--clip', type=str, help='Test on specific clip')
    parser.add_argument('--multi', type=int, default=3, help='Test on N clips')
    parser.add_argument('--compare', action='store_true', help='Compare tracking vs no tracking')
    
    args = parser.parse_args()
    
    if args.clip:
        # Test single clip
        test_tracking_on_clip(args.clip)
    elif args.compare:
        # Comparison test
        clips_dir = Path('inputs/clips')
        clips = list(clips_dir.glob('*.mp4'))
        if clips:
            compare_tracking_vs_no_tracking(clips[0])
        else:
            print("Error: No clips found")
    else:
        # Test multiple clips
        test_multiple_clips(args.multi)


if __name__ == '__main__':
    main()
