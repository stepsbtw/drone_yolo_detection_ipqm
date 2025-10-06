"""
YOLOv11 People Detection Pipeline with Weapon Detection
Processes images and detects people, then detects weapons in person crops.
"""

import cv2
import os
import glob
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from estimation import Camera

try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    PERSON_CLASS_ID = 0
    BOX_COLOR = (0, 255, 0)
    BOX_THICKNESS = 2
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

try:
    from weapon_detector import WeaponDetector
    WEAPON_DETECTION_AVAILABLE = True
except ImportError:
    WEAPON_DETECTION_AVAILABLE = False
    print("Warning: Weapon detection not available. Install required packages: pip install inference supervision python-dotenv")


class DetectionStatistics:
    """Class to track comprehensive detection statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_images = 0
        self.images_with_people = 0
        self.total_people = 0
        self.total_weapons = 0
        self.people_with_weapons = 0
        self.total_samples = 0
        self.samples_with_weapons = 0
        self.current_sample_has_weapons = False
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.prec= 0
        self.recall = 0
        self.f1score = 0
        # Distance tracking
        self.distances = []
        self.people_with_distance = 0
    
    def start_new_sample(self):
        """Mark the start of a new sample directory."""
        if self.current_sample_has_weapons:
            self.samples_with_weapons += 1
        self.current_sample_has_weapons = False
        self.total_samples += 1
    
    def add_image_results(self, num_people, num_weapons, people_with_weapons_count, with_weapons, distances=None):
        """Add results from processing one image."""

        self.total_images += 1
        if num_people > 0:
            self.images_with_people += 1
        
        self.total_people += num_people
        self.total_weapons += num_weapons
        self.people_with_weapons += people_with_weapons_count
        
        if num_weapons > 0:
            self.current_sample_has_weapons = True
            if with_weapons is True:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if with_weapons is True:
                self.fn += 1
            else:
                self.tn += 1
        
        # Add distance information
        if distances:
            self.distances.extend(distances)
            self.people_with_distance += len(distances)
        
        if num_weapons > 0:
            self.current_sample_has_weapons = True
    
    def finalize(self):
        """Finalize statistics (call at the end)."""
        if self.current_sample_has_weapons:
            self.samples_with_weapons += 1
    
    def get_percentages(self):
        """Calculate and return percentage statistics."""
        # Percentage of images with people
        people_in_images_pct = (self.images_with_people / self.total_images * 100) if self.total_images > 0 else 0
        
        # Percentage of people with weapons
        weapons_in_people_pct = (self.people_with_weapons / self.total_people * 100) if self.total_people > 0 else 0
        
        # Percentage of samples with weapons
        weapons_in_samples_pct = (self.samples_with_weapons / self.total_samples * 100) if self.total_samples > 0 else 0
        
        # Calculate metrics with zero division protection
        total_predictions = self.tp + self.tn + self.fp + self.fn
        if total_predictions > 0:
            self.accuracy = (self.tp + self.tn) / total_predictions
        else:
            self.accuracy = 0
            
        if (self.tp + self.fp) > 0:
            self.precision = self.tp / (self.tp + self.fp)
        else:
            self.precision = 0
            
        if (self.tp + self.fn) > 0:
            self.recall = self.tp / (self.tp + self.fn)
        else:
            self.recall = 0
            
        if (self.precision + self.recall) > 0:
            self.f1score = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
        else:
            self.f1score = 0  

        
        avg_distance = np.mean(self.distances) if self.distances else 0
        
        return {
            'people_in_images_pct': people_in_images_pct,
            'weapons_in_people_pct': weapons_in_people_pct,
            'weapons_in_samples_pct': weapons_in_samples_pct,
            'total_images': self.total_images,
            'images_with_people': self.images_with_people,
            'total_people': self.total_people,
            'total_weapons': self.total_weapons,
            'people_with_weapons': self.people_with_weapons,
            'total_samples': self.total_samples,
            'samples_with_weapons': self.samples_with_weapons,
            'accuracy':self.accuracy,
            'precision':self.precision,
            'recall':self.recall,
            'f1score':self.f1score,
            'tp':self.tp,
            'tn':self.tn,
            'fp':self.fp,
            'fn':self.fn,
            
            'people_with_distance': self.people_with_distance,
            'avg_distance': avg_distance,
            'total_distances': len(self.distances)
        }
    
    def print_summary(self):
        """Print comprehensive statistics summary."""
        stats = self.get_percentages()
        
        print("\\n" + "=" * 60)
        print("📊 COMPREHENSIVE DETECTION STATISTICS")
        print("=" * 60)
        
        print(f"\\n📸 IMAGE STATISTICS:")
        print(f"   Total images processed: {stats['total_images']:,}")
        print(f"   Images with people: {stats['images_with_people']:,} ({stats['people_in_images_pct']:.1f}%)")
        
        print(f"\\n👥 PEOPLE STATISTICS:")
        print(f"   Total people detected: {stats['total_people']:,}")
        print(f"   People with weapons: {stats['people_with_weapons']:,} ({stats['weapons_in_people_pct']:.1f}%)")
        
        print(f"\\n🔫 WEAPON STATISTICS:")
        print(f"   Total weapons detected: {stats['total_weapons']:,}")
        
        print(f"\\n📁 SAMPLE STATISTICS:")
        print(f"   Total samples processed: {stats['total_samples']:,}")
        print(f"   Samples with weapons: {stats['samples_with_weapons']:,} ({stats['weapons_in_samples_pct']:.1f}%)")
        
        if stats['total_distances'] > 0:
            print(f"   People with distance data: {stats['people_with_distance']:,}")
            print(f"   Average distance: {stats['avg_distance']:.2f}m")
        else:
            print(f"   No distance data available")
        
        print(f"\\n�� KEY PERCENTAGES:")
        print(f"   🎯 People in images: {stats['people_in_images_pct']:.1f}% of images contain people")
        print(f"   ⚔️  Weapons in people: {stats['weapons_in_people_pct']:.1f}% of people have weapons")
        print(f"   📁 Weapons in samples: {stats['weapons_in_samples_pct']:.1f}% of samples contain weapons")
        
        print(f'\nAccuracy: {stats['accuracy']}')
        print(f'\nPrecision: {stats['precision']}')
        print(f'\nRecall: {stats['recall']}')
        print(f'\nF1-Score: {stats['f1score']}')
        print(f'\nTP: {stats['tp']}')
        print(f'\nTN: {stats['tn']}')
        print(f'\nFP: {stats['fp']}')
        print(f'\nFN: {stats['fn']}')

        if stats['total_weapons'] > 0:
            avg_weapons_per_person = stats['total_weapons'] / stats['people_with_weapons']
            print(f"   📊 Average weapons per armed person: {avg_weapons_per_person:.1f}")
        
        print("\\n" + "=" * 60)


class PeopleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.4, enable_weapon_detection: bool = True):
        """
        Initialize the people detector with YOLOv11 model.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            enable_weapon_detection: Whether to enable weapon detection on person crops
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        # COCO class ID for 'person' is 0
        self.person_class_id = PERSON_CLASS_ID
        self.save_crops = True  # Default to saving crops
        
        # Initialize statistics tracker
        self.stats = DetectionStatistics()
        
        # Initialize weapon detector if available and enabled
        self.weapon_detector = None
        self.enable_weapon_detection = enable_weapon_detection and WEAPON_DETECTION_AVAILABLE
        
        if self.enable_weapon_detection:
            try:
                self.weapon_detector = WeaponDetector()
                print("Weapon detection enabled")
            except Exception as e:
                print(f"Failed to initialize weapon detector: {e}")
                self.enable_weapon_detection = False
        else:
            if enable_weapon_detection and not WEAPON_DETECTION_AVAILABLE:
                print("Weapon detection requested but not available")
            else:
                print("Weapon detection disabled")
        self.camera = None
        
        try:
            self.camera = Camera()  # Using default drone camera settings
            self.distance_logger = logging.getLogger('distance_logger')
            self.distance_logger.setLevel(logging.INFO)
            
            if not self.distance_logger.handlers:
                log_file = 'person_distances.log'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.distance_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Failed to initialize distance estimation: {e}")
    
    def extract_real_distance_from_filename(self, filepath: str):
        try:
            dir_path = os.path.dirname(filepath)
            dir_name = os.path.basename(dir_path)
            
            if not dir_name:
                dir_name = os.path.splitext(os.path.basename(filepath))[0]
            parts = dir_name.split('_')

            for i in range(len(parts) - 1):
                if parts[i + 1].isdigit() and i + 2 < len(parts) and parts[i + 2].isdigit():
                    distance = float(parts[i + 1])
                    return distance
            
            return None
        except (ValueError, IndexError):
            return None
        
    def extract_camera_height_from_filename(self, filepath: str):
        try:
            dir_path = os.path.dirname(filepath)
            dir_name = os.path.basename(dir_path)

            if not dir_name:
                dir_name = os.path.splitext(os.path.basename(filepath))[0]
            parts = dir_name.split('_')

            for i in range(len(parts) - 1):
                if parts[i + 1].isdigit() and i + 2 < len(parts) and parts[i + 2].isdigit():
                    height = float(parts[i + 2])
                    return height
            
            return None
        except (ValueError, IndexError):
            return None
        
    def detect_people(self, image_path: str):
        """
        Detect people in a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            tuple: (image_with_boxes, detections_info)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run inference
        results = self.model(image, imgsz=640, iou=0.6, conf = self.confidence_threshold, verbose=False)
        
        
        # Process results
        detections_info = []
        image_with_boxes = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for person_idx, box in enumerate(boxes):
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process person detections above threshold
                    if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                        # height pixels
                        person_height_px = y2 - y1
                        
                        distance_m = None
                        if self.camera:
                            try:
                                distance_m = self.camera.estimate_distance(person_height_px)
                                
                                # Extract real distance and camera height from file path
                                image_name = os.path.basename(image_path)
                                real_distance_m = self.extract_real_distance_from_filename(image_path)
                                
                                log_message = (f"Image: {image_name}, Person: {person_idx + 1}, "
                                             f"PixelHeight: {person_height_px:.1f}px, "
                                             f"Estimated: {distance_m:.2f}m, "
                                             f"Real: {real_distance_m:.2f}m, " if real_distance_m else f"Real: N/A, "
                                             f"Confidence: {confidence:.3f}, "
                                             f"BBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                                self.distance_logger.info(log_message)
                                
                                # Enhanced console output
                                console_msg = f"  -> Person {person_idx + 1}: Est:{distance_m:.2f}m"
                                if real_distance_m is not None:
                                    console_msg += f", Real:{real_distance_m:.2f}m)"
                                console_msg += f", {person_height_px:.1f}px"
                                print(console_msg)
                                
                            except Exception as e:
                                print(f"Warning: Failed to estimate distance for person {person_idx + 1}: {e}")
                        
                        # Draw bounding box
                        cv2.rectangle(image_with_boxes, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    BOX_COLOR, BOX_THICKNESS)
                        
                        # Add confidence label
                        label = f"Person: {confidence:.2f}"
                        if distance_m is not None:
                            label += f" ({distance_m:.1f}m)"
                        
                        cv2.putText(image_with_boxes, label, 
                                  (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 
                                  BOX_COLOR, FONT_THICKNESS)
                        
                        # Store detection info
                        detection_info = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': 'person',
                            'height_px': float(person_height_px)
                        }
                        
                        # Add distance if available
                        if distance_m is not None:
                            detection_info['distance_m'] = float(distance_m)
                        
                        detections_info.append(detection_info)
        
        return image_with_boxes, detections_info
    
    def extract_person_crops(self, image, detections_info):
        """
        Extract person crops from the original image based on detections.
        
        Args:
            image: Original image (numpy array)
            detections_info: List of detection dictionaries
            
        Returns:
            list: List of tuples (crop_image, crop_info)
        """
        crops = []
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Add padding to the bounding box
            try:
                padding = CROP_PADDING
            except NameError:
                padding = 0.1  # Default fallback
                
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(image.shape[1], x2 + pad_x)
            y2_pad = min(image.shape[0], y2 + pad_y)
            
            # Check minimum size
            crop_width = x2_pad - x1_pad
            crop_height = y2_pad - y1_pad
            
            try:
                min_size = CROP_MIN_SIZE
            except NameError:
                min_size = 32  # Default fallback
                
            #if crop_width < min_size or crop_height < min_size:
            #    continue  # Skip crops that are too small
            
            # Crop the image
            cropped_person = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Store crop info
            crop_info = {
                'person_id': i + 1,
                'bbox': detection['bbox'],
                'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                'confidence': confidence,
                'crop_size': (crop_width, crop_height)
            }
            crops.append((cropped_person, crop_info))
        
        return crops
    
    def detect_weapons_in_crops(self, crops_with_info):
        """
        Detect weapons in person crops using the weapon detector.
        
        Args:
            crops_with_info: List of tuples (crop_image, crop_info)
            
        Returns:
            list: List of weapon detection results
        """
        if not self.enable_weapon_detection or not self.weapon_detector:
            return []
        
        return self.weapon_detector.process_multiple_crops(crops_with_info)
    
    def save_weapon_detection_results(self, weapon_results, output_dir, base_filename):
        """
        Save weapon detection results - only weapon bounding box crops.
        
        Args:
            weapon_results: List of weapon detection results
            output_dir: Output directory
            base_filename: Base filename for saving
            
        Returns:
            tuple: (Number of weapon detections saved, number of people with weapons)
        """
        if not weapon_results:
            return 0, 0
        
        # Create weapon detection output directory
        weapons_dir = os.path.join(output_dir, "weapon_detections")
        Path(weapons_dir).mkdir(parents=True, exist_ok=True)
        
        weapons_detected = 0
        people_with_weapons = 0
        
        for result in weapon_results:
            person_id = result['person_info']['person_id']
            person_confidence = result['person_info']['confidence']
            
            if result['has_weapons'] and result['weapon_crops']:
                people_with_weapons += 1
                
                # Save each weapon crop separately
                for weapon_idx, weapon_crop_info in enumerate(result['weapon_crops']):
                    weapon_crop = weapon_crop_info['crop']
                    weapon_confidence = weapon_crop_info['confidence']
                    weapon_class = weapon_crop_info['class']
                    
                    # Generate filename for weapon crop
                    weapon_filename = f"{base_filename}_person_{person_id:02d}_weapon_{weapon_idx+1:02d}_{weapon_class}_conf_{weapon_confidence:.2f}.jpg"
                    weapon_path = os.path.join(weapons_dir, weapon_filename)
                    
                    # Save the weapon crop
                    cv2.imwrite(weapon_path, weapon_crop)
                    weapons_detected += 1
        
        return weapons_detected, people_with_weapons

    def save_bounding_box_crops(self, image, detections_info, crops_dir, base_filename):
        """
        Save individual cropped images for each detected person.
        
        Args:
            image: Original image (numpy array)
            detections_info: List of detection dictionaries
            crops_dir: Directory to save crops (already created)
            base_filename: Base filename (without extension)
        """
        
        saved_crops = 0
        
        for i, detection in enumerate(detections_info):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Add padding to the bounding box
            try:
                padding = CROP_PADDING
            except NameError:
                padding = 0.1  # Default fallback
                
            width = x2 - x1
            height = y2 - y1
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(image.shape[1], x2 + pad_x)
            y2_pad = min(image.shape[0], y2 + pad_y)
            
            # Check minimum size
            crop_width = x2_pad - x1_pad
            crop_height = y2_pad - y1_pad
            
            try:
                min_size = CROP_MIN_SIZE
            except NameError:
                min_size = 32  # Default fallback
                
            if crop_width < min_size or crop_height < min_size:
                continue  # Skip crops that are too small
            
            # Crop the image
            cropped_person = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Generate filename for the crop
            crop_filename = f"{base_filename}_person_{i+1:02d}_conf_{confidence:.2f}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save the cropped image
            cv2.imwrite(crop_path, cropped_person)
            saved_crops += 1
        
        return saved_crops
    
    def process_directory(self, input_dir: str, output_dir: str, with_weapons=False):
        """
        Process all images in a directory and save results.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
        """
        # Create organized output directories
        detections_dir = os.path.join(output_dir, "detections")
        crops_dir = os.path.join(output_dir, "crops")
        
        Path(detections_dir).mkdir(parents=True, exist_ok=True)
        if self.save_crops:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for file in os.listdir(input_dir):
            if os.path.splitext(file)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]:
                image_files.append(os.path.join(input_dir, file))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                #print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Load original image for cropping
                original_image = cv2.imread(image_path)
                
                # Detect people
                image_with_boxes, detections = self.detect_people(image_path)
                
                # Save result with bounding boxes to detections folder
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_detected{ext}"
                detection_path = os.path.join(detections_dir, output_filename)
                
                cv2.imwrite(detection_path, image_with_boxes)
                
                # Extract person crops for weapon detection
                person_crops = []
                crops_saved = 0
                weapons_detected = 0
                people_with_weapons_count = 0
                
                if detections:
                    # Extract person crops
                    person_crops = self.extract_person_crops(original_image, detections)
                    
                    # Save individual bounding box crops to crops folder
                    if self.save_crops:
                        crops_saved = self.save_bounding_box_crops(original_image, detections, crops_dir, name)
                    
                    # Perform weapon detection on person crops
                    if self.enable_weapon_detection and person_crops:
                        #print(f"  -> Checking {len(person_crops)} person crops for weapons...")
                        weapon_results = self.detect_weapons_in_crops(person_crops)
                        weapons_detected, people_with_weapons_count = self.save_weapon_detection_results(weapon_results, output_dir, name)
                
                # Extract distances from detections
                distances = [d['distance_m'] for d in detections if 'distance_m' in d]
                
                # Update statistics
                self.stats.add_image_results(len(detections), weapons_detected, people_with_weapons_count, with_weapons, distances)
                
                # Print detection summary
                if detections:
                    summary_parts = [f"Found {len(detections)} people"]
                    if self.save_crops:
                        summary_parts.append(f"saved {crops_saved} crops")
                    if self.enable_weapon_detection:
                        summary_parts.append(f"detected {weapons_detected} weapons")
                        if people_with_weapons_count > 0:
                            summary_parts.append(f"({people_with_weapons_count} people with weapons)")
                    #print(f"  -> {', '.join(summary_parts)}")
                else:
                    print(f"  -> No people detected, saved full image only")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Print directory summary for single directory processing
        if not hasattr(self.stats, '_in_batch_mode'):
            print(f"\\nDirectory processing complete!")
            stats = self.stats.get_percentages()
            print(f"Images processed: {stats['total_images']}")
            print(f"People detected: {stats['total_people']}")
            if self.enable_weapon_detection:
                print(f"Weapons detected: {stats['total_weapons']}")
                if stats['total_people'] > 0:
                    print(f"Weapon rate: {stats['weapons_in_people_pct']:.1f}% of people have weapons")
            if stats['total_distances'] > 0:
                print(f"Distance measurements: {stats['total_distances']}")
                print(f"Average distance: {stats['avg_distance']:.2f}m")
    
    def process_all_sample_directories(self, samples_dir: str, output_base_dir: str, with_weapons):
        """
        Process all sample directories and maintain organized folder structure.
        
        Args:
            samples_dir: Base directory containing sample folders
            output_base_dir: Base output directory
        """
        # Get all subdirectories in samples
        sample_dirs = [d for d in os.listdir(samples_dir) 
                      if os.path.isdir(os.path.join(samples_dir, d))]
        
        print(f"Found {len(sample_dirs)} sample directories")
        
        # Create main organized output structure
        detections_base_dir = os.path.join(output_base_dir, "detections")
        crops_base_dir = os.path.join(output_base_dir, "crops")
        
        # Reset statistics for batch processing
        #self.stats.reset()
        self.stats._in_batch_mode = True  # Flag to indicate batch processing
        
        for sample_idx, sample_dir in enumerate(sample_dirs, 1):
            input_path = os.path.join(samples_dir, sample_dir)
            
            # Create organized output paths for this sample directory
            sample_detections_dir = os.path.join(detections_base_dir, sample_dir)
            sample_crops_dir = os.path.join(crops_base_dir, sample_dir)
            
            # Create temporary output structure for this sample
            temp_output = os.path.join(output_base_dir, "temp", sample_dir)
            
            print(f"\n Processing sample {sample_idx}/{len(sample_dirs)}: {sample_dir}")
            
            # Mark start of new sample for statistics
            self.stats.start_new_sample()
            
            self.process_directory(input_path, temp_output, with_weapons)
            
            # Move results to organized structure
            self._organize_sample_output(temp_output, sample_detections_dir, sample_crops_dir)
        
        # Finalize statistics
        #self.stats.finalize()
        
        # Clean up empty weapon detection directories
        self._cleanup_empty_weapon_directories(output_base_dir)
   
    
        


    def _organize_sample_output(self, temp_dir: str, detections_dir: str, crops_dir: str):
        import os    
        import shutil
        from pathlib import Path
        import stat
        
        """
        Move processed files from temp directory to organized structure.
        Handles Windows permission issues.
        """

        def on_rm_error(func, path, exc_info):
            """Force delete read-only files (Windows quirk)."""
            os.chmod(path, stat.S_IWRITE)
            func(path)

        temp_detections = os.path.join(temp_dir, "detections")
        temp_crops = os.path.join(temp_dir, "crops")
        temp_weapons = os.path.join(temp_dir, "weapon_detections")

        # Move detections
        if os.path.exists(temp_detections):
            Path(detections_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_detections):
                src = os.path.join(temp_detections, file)
                dst = os.path.join(detections_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Move crops
        if os.path.exists(temp_crops) and getattr(self, "save_crops", True):
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_crops):
                src = os.path.join(temp_crops, file)
                dst = os.path.join(crops_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Move weapon detections
        if os.path.exists(temp_weapons):
            weapons_base_dir = os.path.dirname(crops_dir).replace("\\crops", "\\weapon_detections").replace("/crops", "/weapon_detections")
            weapons_dir = os.path.join(weapons_base_dir, os.path.basename(crops_dir))
            Path(weapons_dir).mkdir(parents=True, exist_ok=True)
            for file in os.listdir(temp_weapons):
                src = os.path.join(temp_weapons, file)
                dst = os.path.join(weapons_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Clean up temp directory (safe version)
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, onerror=on_rm_error)
            except PermissionError:
                print(f"[WARNING] Could not remove {temp_dir} due to permission issues.")

    
    def _cleanup_empty_weapon_directories(self, output_base_dir: str):
        """
        Remove empty weapon detection directories after processing is complete.
        """
        weapons_base_dir = os.path.join(output_base_dir, "weapon_detections")
        
        if not os.path.exists(weapons_base_dir):
            return
        
        # Find and remove empty directories
        empty_dirs = []
        for root, dirs, files in os.walk(weapons_base_dir, topdown=False):
            # Skip the base weapon_detections directory itself
            if root == weapons_base_dir:
                continue
                
            # Check if directory is empty (no files)
            if not files:
                empty_dirs.append(root)
        
        # Remove empty directories
        removed_count = 0
        for empty_dir in empty_dirs:
            try:
                os.rmdir(empty_dir)
                removed_count += 1
            except OSError:
                # Directory might not be empty or have permission issues
                pass
        
        if removed_count > 0:
            print(f"\n🧹 Cleaned up {removed_count} empty weapon detection directories")
