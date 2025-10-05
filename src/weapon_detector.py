"""
Weapon Detection using Roboflow Model
Detects weapons in cropped person images.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from inference import get_model
import supervision as sv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeaponDetector:
    #https://universe.roboflow.com/weapon-rcjrw/weapon-detection-pgqnr/model/3
    def __init__(self, model_id: str = "weapon-detection-pgqnr/3", confidence_threshold: float = 0.2):
        """
        Initialize the weapon detector with Roboflow model.
        
        Args:
            model_id: Roboflow model ID for weapon detection
            confidence_threshold: Minimum confidence for detections
        """
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        
        # Set up API key from environment
        self.api_key = 'AyaEAmTzmir20T8S42Dm' #os.getenv('ROBOFLOW_API_KEY') or os.getenv('roboflow')
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not found in environment variables (.env file)")
        
        # Set the API key in environment for inference library
        os.environ['ROBOFLOW_API_KEY'] = self.api_key
        
        # Load the model
        try:
            self.model = get_model(model_id=self.model_id, api_key = self.api_key)
            print(f"Loaded weapon detection model: {self.model_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to load weapon detection model: {e}")
    
    def detect_weapons(self, image):
        """
        Detect weapons in a single image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            tuple: (annotated_image, detections_info, weapon_crops)
        """
        try:
            # Run inference
            results = self.model.infer(image, overlap = 0.50, confidence = self.confidence_threshold)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_inference(results)
            
               
            # Create annotated image
            annotated_image = image.copy()
            
            # Extract weapon crops
            weapon_crops = []
            
            if len(detections) > 0:
                # Create supervision annotators
                bounding_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
                label_annotator = sv.LabelAnnotator(color=sv.Color.RED)
                
                # Create labels with confidence scores
                labels = []
                for i in range(len(detections)):
                    class_name = detections.data.get('class_name', ['weapon'])[i] if 'class_name' in detections.data else 'weapon'
                    confidence = detections.confidence[i]
                    labels.append(f"{class_name}: {confidence:.2f}")
                
                # Annotate the image
                annotated_image = bounding_box_annotator.annotate(
                    scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels)
                
                # Extract weapon crops from detections
                for i in range(len(detections)):
                    bbox = detections.xyxy[i]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Add small padding to weapon bbox
                    padding = 5
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding)
                    x2_pad = min(image.shape[1], x2 + padding)
                    y2_pad = min(image.shape[0], y2 + padding)
                    
                    # Extract weapon crop
                    weapon_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if weapon_crop.size > 0:  # Ensure crop is valid
                        weapon_crops.append({
                            'crop': weapon_crop,
                            'bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
                            'confidence': float(detections.confidence[i]),
                            'class': detections.data.get('class_name', ['weapon'])[i] if 'class_name' in detections.data else 'weapon'
                        })
            
            # Extract detection info
            detections_info = []
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_name = detections.data.get('class_name', ['weapon'])[i] if 'class_name' in detections.data else 'weapon'
                
                detections_info.append({
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'confidence': float(confidence),
                    'class': class_name
                })
            
            return annotated_image, detections_info, weapon_crops
            
        except Exception as e:
            print(f"Error in weapon detection: {e}")
            return image, [], []
    
    def process_person_crop(self, crop_image, crop_info):
        """
        Process a single person crop for weapon detection.
        
        Args:
            crop_image: Cropped person image (numpy array)
            crop_info: Information about the person crop
            
        Returns:
            dict: Results containing annotated image and detection info
        """
        annotated_crop, weapon_detections, weapon_crops = self.detect_weapons(crop_image)
        
        return {
            'original_crop': crop_image,
            'annotated_crop': annotated_crop,
            'person_info': crop_info,
            'weapon_detections': weapon_detections,
            'weapon_crops': weapon_crops,
            'has_weapons': len(weapon_detections) > 0
        }
    
    def process_multiple_crops(self, crops_with_info):
        """
        Process multiple person crops for weapon detection.
        
        Args:
            crops_with_info: List of tuples (crop_image, crop_info)
            
        Returns:
            list: List of detection results for each crop
        """
        results = []
        
        for crop_image, crop_info in crops_with_info:
            result = self.process_person_crop(crop_image, crop_info)
            results.append(result)
            
            # Log detection results
            # if result['has_weapons']:
            #     weapon_count = len(result['weapon_detections'])
            #     print(f"  -> Weapons detected in person crop: {weapon_count} weapon(s)")
            # else:
            #     print(f"  -> No weapons detected in person crop")
        
        return results
