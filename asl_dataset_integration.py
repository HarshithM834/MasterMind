"""
ASL Dataset Integration System
Integrates multiple large-scale ASL datasets for maximum accuracy and precision
"""

import os
import json
import requests
import zipfile
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp
from collections import defaultdict
import logging

class ASLDatasetIntegration:
    def __init__(self, data_dir="asl_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe for feature extraction
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Dataset configurations
        self.datasets = {
            'kaggle_asl': {
                'name': 'Kaggle ASL Dataset',
                'url': 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet',
                'description': 'Large ASL alphabet dataset with 87,000+ images',
                'format': 'images',
                'classes': 29  # 26 letters + space + delete + nothing
            },
            'asl_letters': {
                'name': 'ASL Letters Dataset',
                'url': 'https://www.kaggle.com/datasets/ayuraj/asl-dataset',
                'description': 'Comprehensive ASL letters dataset',
                'format': 'images',
                'classes': 26
            },
            'hand_gesture_recognition': {
                'name': 'Hand Gesture Recognition Dataset',
                'url': 'https://www.kaggle.com/datasets/kuchhbhi/hand-gesture-recognition-dataset',
                'description': 'Multi-class hand gesture dataset',
                'format': 'images',
                'classes': 10
            }
        }
        
        # ASL letter mappings
        self.asl_mappings = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
            7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
            14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
            21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
        }
        
        self.reverse_mappings = {v: k for k, v in self.asl_mappings.items()}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def download_kaggle_dataset(self, dataset_name, kaggle_username=None, kaggle_key=None):
        """Download dataset from Kaggle using API"""
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset
            dataset_path = f"grassknoted/{dataset_name}"
            api.dataset_download_files(dataset_path, path=self.data_dir, unzip=True)
            
            self.logger.info(f"Successfully downloaded {dataset_name} from Kaggle")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download from Kaggle: {e}")
            return False
    
    def create_synthetic_dataset(self):
        """Create synthetic ASL dataset using MediaPipe for data augmentation"""
        self.logger.info("Creating synthetic ASL dataset...")
        
        synthetic_dir = self.data_dir / "synthetic_asl"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Create directories for each letter
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            (synthetic_dir / letter).mkdir(exist_ok=True)
        
        # Generate synthetic hand poses for each letter
        for letter, class_id in self.reverse_mappings.items():
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                self.generate_synthetic_poses(letter, synthetic_dir / letter, num_samples=1000)
        
        self.logger.info("Synthetic dataset creation completed")
    
    def generate_synthetic_poses(self, letter, output_dir, num_samples=1000):
        """Generate synthetic hand poses for a specific ASL letter"""
        # Define hand landmark templates for each letter
        letter_templates = self.get_asl_letter_templates()
        
        if letter not in letter_templates:
            return
        
        template = letter_templates[letter]
        
        for i in range(num_samples):
            # Add noise and variations to the template
            noisy_template = self.add_pose_variations(template)
            
            # Create image from landmarks
            image = self.landmarks_to_image(noisy_template)
            
            # Save the image
            filename = f"{letter}_{i:04d}.jpg"
            cv2.imwrite(str(output_dir / filename), image)
    
    def get_asl_letter_templates(self):
        """Define hand landmark templates for each ASL letter"""
        templates = {
            'A': self.create_fist_template(),
            'B': self.create_open_hand_template(),
            'C': self.create_c_shape_template(),
            'D': self.create_pointing_template(),
            'E': self.create_all_fingers_template(),
            'F': self.create_ok_sign_template(),
            'G': self.create_pointing_template(),
            'H': self.create_two_fingers_close_template(),
            'I': self.create_pinky_template(),
            'J': self.create_hook_template(),
            'K': self.create_two_fingers_apart_template(),
            'L': self.create_l_shape_template(),
            'M': self.create_three_fingers_template(),
            'N': self.create_two_fingers_template(),
            'O': self.create_circle_template(),
            'P': self.create_pointing_down_template(),
            'Q': self.create_pointing_side_template(),
            'R': self.create_crossed_fingers_template(),
            'S': self.create_fist_thumb_over_template(),
            'T': self.create_thumb_between_template(),
            'U': self.create_two_fingers_template(),
            'V': self.create_two_fingers_template(),
            'W': self.create_three_fingers_template(),
            'X': self.create_bent_finger_template(),
            'Y': self.create_thumb_pinky_template(),
            'Z': self.create_pointing_template()
        }
        return templates
    
    def create_fist_template(self):
        """Create landmark template for fist (A)"""
        landmarks = []
        # Simplified landmark positions for fist
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.01),
                'y': 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_open_hand_template(self):
        """Create landmark template for open hand (B)"""
        landmarks = []
        # Simplified landmark positions for open hand
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [4, 8, 12, 16, 20] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_c_shape_template(self):
        """Create landmark template for C shape"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.4 + np.random.normal(0, 0.02) if i in [8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_pointing_template(self):
        """Create landmark template for pointing (D, G, Z)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i == 8 else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_all_fingers_template(self):
        """Create landmark template for all fingers extended (E)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.2 + np.random.normal(0, 0.02) if i in [4, 8, 12, 16, 20] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_ok_sign_template(self):
        """Create landmark template for OK sign (F)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [4, 8] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_two_fingers_close_template(self):
        """Create landmark template for two fingers close (H)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_pinky_template(self):
        """Create landmark template for pinky extended (I)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i == 20 else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_hook_template(self):
        """Create landmark template for hook (J)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i == 20 else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_two_fingers_apart_template(self):
        """Create landmark template for two fingers apart (K)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_l_shape_template(self):
        """Create landmark template for L shape"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [4, 8] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_three_fingers_template(self):
        """Create landmark template for three fingers (M, W)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [8, 12, 16] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_two_fingers_template(self):
        """Create landmark template for two fingers (N, U, V)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_circle_template(self):
        """Create landmark template for circle (O)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.5 + np.random.normal(0, 0.02),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_pointing_down_template(self):
        """Create landmark template for pointing down (P)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.7 + np.random.normal(0, 0.02) if i == 8 else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_pointing_side_template(self):
        """Create landmark template for pointing side (Q)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.7 + np.random.normal(0, 0.02) if i == 8 else 0.5 + np.random.normal(0, 0.01),
                'y': 0.5 + np.random.normal(0, 0.02),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_crossed_fingers_template(self):
        """Create landmark template for crossed fingers (R)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_fist_thumb_over_template(self):
        """Create landmark template for fist with thumb over (S)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.5 + np.random.normal(0, 0.02),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_thumb_between_template(self):
        """Create landmark template for thumb between fingers (T)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [4, 8, 12] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_bent_finger_template(self):
        """Create landmark template for bent finger (X)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.5 + np.random.normal(0, 0.02),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def create_thumb_pinky_template(self):
        """Create landmark template for thumb and pinky (Y)"""
        landmarks = []
        for i in range(21):
            landmark = type('Landmark', (), {
                'x': 0.5 + np.random.normal(0, 0.02),
                'y': 0.3 + np.random.normal(0, 0.02) if i in [4, 20] else 0.5 + np.random.normal(0, 0.01),
                'z': np.random.normal(0, 0.01)
            })()
            landmarks.append(landmark)
        return landmarks
    
    def add_pose_variations(self, template):
        """Add realistic variations to hand pose template"""
        variations = []
        for landmark in template:
            # Add small random variations
            variation = type('Landmark', (), {
                'x': landmark.x + np.random.normal(0, 0.01),
                'y': landmark.y + np.random.normal(0, 0.01),
                'z': landmark.z + np.random.normal(0, 0.005)
            })()
            variations.append(variation)
        return variations
    
    def landmarks_to_image(self, landmarks, image_size=(224, 224)):
        """Convert landmarks to image representation"""
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Draw hand skeleton
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # Index
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
            [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
            [5, 9], [9, 13], [13, 17]  # Palm
        ]
        
        for connection in connections:
            pt1 = (int(landmarks[connection[0]].x * image_size[1]), 
                   int(landmarks[connection[0]].y * image_size[0]))
            pt2 = (int(landmarks[connection[1]].x * image_size[1]), 
                   int(landmarks[connection[1]].y * image_size[0]))
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            pt = (int(landmark.x * image_size[1]), int(landmark.y * image_size[0]))
            cv2.circle(image, pt, 3, (0, 255, 0), -1)
        
        return image
    
    def load_and_preprocess_dataset(self, dataset_path):
        """Load and preprocess dataset from directory"""
        self.logger.info(f"Loading dataset from {dataset_path}")
        
        images = []
        labels = []
        
        dataset_path = Path(dataset_path)
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name in self.reverse_mappings:
                    class_id = self.reverse_mappings[class_name]
                    
                    for img_file in class_dir.glob("*.jpg"):
                        try:
                            image = cv2.imread(str(img_file))
                            if image is not None:
                                # Resize image
                                image = cv2.resize(image, (224, 224))
                                images.append(image)
                                labels.append(class_id)
                        except Exception as e:
                            self.logger.warning(f"Failed to load {img_file}: {e}")
        
        return np.array(images), np.array(labels)
    
    def extract_mediapipe_features(self, images):
        """Extract MediaPipe hand landmarks from images"""
        features = []
        
        for image in images:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                feature_vector = self.extract_advanced_features(landmarks)
                features.append(feature_vector)
            else:
                # Create zero vector if no hand detected
                features.append(np.zeros(126))
        
        return np.array(features)
    
    def extract_advanced_features(self, landmarks):
        """Extract 126 features from hand landmarks (same as main detector)"""
        if landmarks is None or len(landmarks) != 21:
            return np.zeros(126, dtype=np.float32)
        
        features = []
        
        # 1. Raw landmark coordinates (normalized) - 21 landmarks * 3 coordinates = 63 features
        wrist = landmarks[0]
        for landmark in landmarks:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        # 2. Distances between key points - 6C2 = 15 distances
        key_points = [0, 4, 8, 12, 16, 20]
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                p1 = landmarks[key_points[i]]
                p2 = landmarks[key_points[j]]
                distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                features.append(distance)
        
        # 3. Finger extension ratios - 5 fingers = 5 features
        tips = [4, 8, 12, 16, 20]
        mcp = [2, 5, 9, 13, 17]
        pip = [3, 6, 10, 14, 18]
        
        for i in range(5):
            tip = landmarks[tips[i]]
            mcp_joint = landmarks[mcp[i]]
            pip_joint = landmarks[pip[i]]
            
            if i == 0:  # Thumb
                ratio = abs(tip.x - mcp_joint.x) / (abs(tip.y - mcp_joint.y) + 1e-6)
            else:
                ratio = abs(tip.y - mcp_joint.y) / (abs(pip_joint.y - mcp_joint.y) + 1e-6)
            features.append(ratio)
        
        # 4. Hand orientation and pose - 2 features
        palm_points = [0, 5, 9, 13, 17]
        palm_center_x = np.mean([landmarks[i].x for i in palm_points])
        palm_center_y = np.mean([landmarks[i].y for i in palm_points])
        features.extend([palm_center_x, palm_center_y])
        
        # 5. Finger angles - 4 fingers (skip thumb) = 4 features
        for i in range(1, 5):
            tip = landmarks[tips[i]]
            pip = landmarks[pip[i]]
            mcp = landmarks[mcp[i]]
            
            vec1 = [tip.x - pip.x, tip.y - pip.y]
            vec2 = [pip.x - mcp.x, pip.y - mcp.y]
            
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            norm1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
            norm2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1, 1))
                features.append(angle)
            else:
                features.append(0)
        
        # 6. Hand size and proportions - 3 features
        hand_width = abs(landmarks[5].x - landmarks[17].x)
        hand_height = abs(landmarks[0].y - landmarks[12].y)
        features.extend([hand_width, hand_height, hand_width / (hand_height + 1e-6)])
        
        # 7. Additional geometric features to reach exactly 126
        palm_center = np.array([palm_center_x, palm_center_y])
        for i in range(5):
            tip = np.array([landmarks[tips[i]].x, landmarks[tips[i]].y])
            distance = np.linalg.norm(tip - palm_center)
            features.append(distance)
        
        # Inter-finger distances (10 combinations)
        finger_tips = [4, 8, 12, 16, 20]
        for i in range(5):
            for j in range(i + 1, 5):
                p1 = landmarks[finger_tips[i]]
                p2 = landmarks[finger_tips[j]]
                distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                features.append(distance)
        
        # Hand orientation angle
        wrist_to_middle = np.array([landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y])
        orientation_angle = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
        features.append(orientation_angle)
        
        # Finger spread angles (4 angles between adjacent fingers)
        for i in range(4):
            tip1 = np.array([landmarks[finger_tips[i]].x, landmarks[finger_tips[i]].y])
            tip2 = np.array([landmarks[finger_tips[i+1]].x, landmarks[finger_tips[i+1]].y])
            palm_center_array = np.array([palm_center_x, palm_center_y])
            
            vec1 = tip1 - palm_center_array
            vec2 = tip2 - palm_center_array
            
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features.append(angle)
            else:
                features.append(0)
        
        # Additional hand shape features
        for i in range(5):
            tip = landmarks[tips[i]]
            pip = landmarks[pip[i]]
            mcp = landmarks[mcp[i]]
            
            if i == 0:  # Thumb
                curvature = abs(tip.x - mcp.x) - abs(pip.x - mcp.x)
            else:
                curvature = abs(tip.y - mcp.y) - abs(pip.y - mcp.y)
            features.append(curvature)
        
        # Hand openness
        palm_points_array = np.array([[landmarks[i].x, landmarks[i].y] for i in palm_points])
        palm_center_array = np.mean(palm_points_array, axis=0)
        
        for i in range(5):
            tip = np.array([landmarks[tips[i]].x, landmarks[tips[i]].y])
            distance = np.linalg.norm(tip - palm_center_array)
            features.append(distance)
        
        # Ensure exactly 126 features
        features = features[:126]
        while len(features) < 126:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def create_data_generator(self, images, labels, batch_size=32, augment=True):
        """Create data generator with augmentation"""
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,  # Don't flip ASL signs
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()
        
        return datagen.flow(images, labels, batch_size=batch_size)
    
    def prepare_training_data(self):
        """Prepare comprehensive training data from multiple sources"""
        self.logger.info("Preparing comprehensive training data...")
        
        all_images = []
        all_labels = []
        
        # Load synthetic dataset
        synthetic_path = self.data_dir / "synthetic_asl"
        if synthetic_path.exists():
            images, labels = self.load_and_preprocess_dataset(synthetic_path)
            all_images.extend(images)
            all_labels.extend(labels)
            self.logger.info(f"Loaded {len(images)} synthetic samples")
        
        # Load any existing real datasets
        for dataset_name, config in self.datasets.items():
            dataset_path = self.data_dir / dataset_name
            if dataset_path.exists():
                images, labels = self.load_and_preprocess_dataset(dataset_path)
                all_images.extend(images)
                all_labels.extend(labels)
                self.logger.info(f"Loaded {len(images)} samples from {dataset_name}")
        
        if not all_images:
            self.logger.warning("No datasets found. Creating synthetic dataset...")
            self.create_synthetic_dataset()
            images, labels = self.load_and_preprocess_dataset(self.data_dir / "synthetic_asl")
            all_images.extend(images)
            all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        # Extract MediaPipe features
        self.logger.info("Extracting MediaPipe features...")
        features = self.extract_mediapipe_features(all_images)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        self.logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        info = {
            'available_datasets': [],
            'total_samples': 0,
            'classes': list(self.asl_mappings.values())
        }
        
        for dataset_name, config in self.datasets.items():
            dataset_path = self.data_dir / dataset_name
            if dataset_path.exists():
                samples = sum(1 for _ in dataset_path.rglob("*.jpg"))
                info['available_datasets'].append({
                    'name': config['name'],
                    'path': str(dataset_path),
                    'samples': samples,
                    'description': config['description']
                })
                info['total_samples'] += samples
        
        return info

if __name__ == "__main__":
    # Initialize dataset integration
    dataset_integration = ASLDatasetIntegration()
    
    # Create synthetic dataset
    dataset_integration.create_synthetic_dataset()
    
    # Prepare training data
    X_train, X_test, y_train, y_test = dataset_integration.prepare_training_data()
    
    # Get dataset info
    info = dataset_integration.get_dataset_info()
    print("Dataset Integration Complete!")
    print(f"Total samples: {info['total_samples']}")
    print(f"Available datasets: {len(info['available_datasets'])}")
    print(f"Classes: {info['classes']}")
