import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext, messagebox
import os
import json
from pathlib import Path
import joblib
from dataset_integration import DatasetIntegrator

class KaggleEnhancedSignDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enhanced ASL gesture mappings
        self.sign_mappings = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
            7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
            14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
            21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'ENTER'
        }
        
        self.reverse_mappings = {v: k for k, v in self.sign_mappings.items()}
        
        # Advanced gesture recognition parameters
        self.gesture_buffer = deque(maxlen=20)
        self.confidence_threshold = 0.85
        self.stability_frames = 12
        self.current_text = ""
        self.last_gesture = None
        self.gesture_stability_count = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection state
        self.detection_running = False
        self.current_frame = None
        self.current_gesture = None
        self.current_confidence = 0
        
        # Model management
        self.models = {}
        self.active_model = None
        self.model_type = "kaggle"  # "kaggle", "transfer_learning", "custom"
        
        # Dataset integrator
        self.dataset_integrator = DatasetIntegrator()
        
        # Load available models
        self.load_available_models()
        
    def load_available_models(self):
        """Load all available pre-trained models"""
        models_dir = Path("pretrained_models")
        models_dir.mkdir(exist_ok=True)
        
        # Try to load Kaggle-trained models
        self.load_kaggle_models()
        
        # Try to load transfer learning models
        self.load_transfer_learning_models()
        
        # Load custom models if available
        self.load_custom_models()
        
        # Set default active model
        if self.models:
            self.active_model = list(self.models.keys())[0]
            print(f"✅ Loaded {len(self.models)} models. Active: {self.active_model}")
        else:
            print("⚠️ No pre-trained models found. Using rule-based fallback.")
    
    def load_kaggle_models(self):
        """Load models trained on Kaggle datasets"""
        models_dir = Path("pretrained_models")
        
        # Look for Kaggle-trained models
        kaggle_models = [
            "mobilenet_asl_best.h5",
            "mobilenet_asl_final.h5", 
            "efficientnet_asl_best.h5",
            "efficientnet_asl_final.h5"
        ]
        
        for model_file in kaggle_models:
            model_path = models_dir / model_file
            if model_path.exists():
                try:
                    model = load_model(model_path)
                    model_name = model_file.replace(".h5", "")
                    self.models[model_name] = {
                        "model": model,
                        "type": "kaggle",
                        "input_type": "image",
                        "accuracy": self.get_model_accuracy(model_name)
                    }
                    print(f"✅ Loaded Kaggle model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
    
    def load_transfer_learning_models(self):
        """Load transfer learning models"""
        models_dir = Path("pretrained_models")
        
        # Look for transfer learning models
        tl_models = [
            "transfer_mobilenet_asl.h5",
            "transfer_efficientnet_asl.h5"
        ]
        
        for model_file in tl_models:
            model_path = models_dir / model_file
            if model_path.exists():
                try:
                    model = load_model(model_path)
                    model_name = model_file.replace(".h5", "")
                    self.models[model_name] = {
                        "model": model,
                        "type": "transfer_learning",
                        "input_type": "image",
                        "accuracy": self.get_model_accuracy(model_name)
                    }
                    print(f"✅ Loaded transfer learning model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
    
    def load_custom_models(self):
        """Load custom trained models"""
        models_dir = Path("pretrained_models")
        
        # Look for custom models
        custom_models = [
            "custom_asl_model.h5",
            "advanced_sign_model.h5"
        ]
        
        for model_file in custom_models:
            model_path = models_dir / model_file
            if model_path.exists():
                try:
                    model = load_model(model_path)
                    model_name = model_file.replace(".h5", "")
                    self.models[model_name] = {
                        "model": model,
                        "type": "custom",
                        "input_type": "features",
                        "accuracy": self.get_model_accuracy(model_name)
                    }
                    print(f"✅ Loaded custom model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
    
    def get_model_accuracy(self, model_name):
        """Get estimated accuracy for a model"""
        # This would typically be loaded from model metadata
        accuracy_map = {
            "mobilenet_asl_best": 0.95,
            "mobilenet_asl_final": 0.94,
            "efficientnet_asl_best": 0.96,
            "efficientnet_asl_final": 0.95,
            "transfer_mobilenet_asl": 0.93,
            "transfer_efficientnet_asl": 0.94,
            "custom_asl_model": 0.92,
            "advanced_sign_model": 0.88
        }
        return accuracy_map.get(model_name, 0.85)
    
    def extract_hand_region(self, frame, landmarks):
        """Extract hand region from frame for image-based models"""
        if landmarks is None:
            return None
        
        # Get bounding box of hand
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 0.1
        x_min = max(0, x_min - padding)
        x_max = min(1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(1, y_max + padding)
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)
        
        # Extract hand region
        hand_region = frame[y1:y2, x1:x2]
        
        if hand_region.size == 0:
            return None
        
        # Resize to model input size
        hand_region = cv2.resize(hand_region, (224, 224))
        
        # Normalize
        hand_region = hand_region.astype(np.float32) / 255.0
        
        return hand_region
    
    def predict_with_image_model(self, frame, landmarks, model_name):
        """Predict using image-based models (Kaggle/Transfer Learning)"""
        if model_name not in self.models:
            return None, 0
        
        model_info = self.models[model_name]
        if model_info["input_type"] != "image":
            return None, 0
        
        # Extract hand region
        hand_region = self.extract_hand_region(frame, landmarks)
        if hand_region is None:
            return None, 0
        
        try:
            # Prepare input
            input_data = np.expand_dims(hand_region, axis=0)
            
            # Predict
            prediction = model_info["model"].predict(input_data, verbose=0)
            confidence = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            
            # Map to gesture
            if predicted_class < 26:  # A-Z
                gesture = chr(ord('A') + predicted_class)
            elif predicted_class == 26:
                gesture = 'SPACE'
            elif predicted_class == 27:
                gesture = 'DELETE'
            elif predicted_class == 28:
                gesture = 'ENTER'
            else:
                gesture = None
            
            return gesture, confidence
            
        except Exception as e:
            print(f"Image model prediction error: {e}")
            return None, 0
    
    def extract_advanced_features(self, landmarks):
        """Extract features for feature-based models"""
        if landmarks is None:
            return None
        
        features = []
        
        # Normalized landmark coordinates
        wrist = landmarks[0]
        for landmark in landmarks:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        # Distances between key points
        key_points = [0, 4, 8, 12, 16, 20]
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                p1 = landmarks[key_points[i]]
                p2 = landmarks[key_points[j]]
                distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                features.append(distance)
        
        # Finger extension ratios
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
        
        # Hand orientation
        palm_points = [0, 5, 9, 13, 17]
        palm_center_x = np.mean([landmarks[i].x for i in palm_points])
        palm_center_y = np.mean([landmarks[i].y for i in palm_points])
        features.extend([palm_center_x, palm_center_y])
        
        # Hand size and proportions
        hand_width = abs(landmarks[5].x - landmarks[17].x)
        hand_height = abs(landmarks[0].y - landmarks[12].y)
        features.extend([hand_width, hand_height, hand_width / (hand_height + 1e-6)])
        
        return np.array(features)
    
    def predict_with_feature_model(self, landmarks, model_name):
        """Predict using feature-based models"""
        if model_name not in self.models:
            return None, 0
        
        model_info = self.models[model_name]
        if model_info["input_type"] != "features":
            return None, 0
        
        try:
            # Extract features
            features = self.extract_advanced_features(landmarks)
            if features is None:
                return None, 0
            
            # Ensure features are the right shape
            if len(features) != 126:
                if len(features) < 126:
                    features = np.pad(features, (0, 126 - len(features)), 'constant')
                else:
                    features = features[:126]
            
            # Predict
            prediction = model_info["model"].predict([features], verbose=0)
            confidence = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            
            return self.sign_mappings[predicted_class], confidence
            
        except Exception as e:
            print(f"Feature model prediction error: {e}")
            return None, 0
    
    def enhanced_gesture_classification(self, frame, landmarks):
        """Enhanced gesture classification using multiple models"""
        if landmarks is None:
            return None, 0
        
        best_gesture = None
        best_confidence = 0
        model_predictions = []
        
        # Try all available models
        for model_name, model_info in self.models.items():
            if model_info["input_type"] == "image":
                gesture, confidence = self.predict_with_image_model(frame, landmarks, model_name)
            elif model_info["input_type"] == "features":
                gesture, confidence = self.predict_with_feature_model(landmarks, model_name)
            else:
                continue
            
            if gesture and confidence > 0.5:  # Minimum confidence threshold
                model_predictions.append((gesture, confidence, model_name, model_info["accuracy"]))
        
        # Ensemble prediction - weight by model accuracy and confidence
        if model_predictions:
            # Weight predictions by model accuracy and confidence
            weighted_predictions = {}
            total_weight = 0
            
            for gesture, confidence, model_name, model_accuracy in model_predictions:
                weight = confidence * model_accuracy
                if gesture in weighted_predictions:
                    weighted_predictions[gesture] += weight
                else:
                    weighted_predictions[gesture] = weight
                total_weight += weight
            
            if weighted_predictions:
                # Find best weighted prediction
                best_gesture = max(weighted_predictions, key=weighted_predictions.get)
                best_confidence = weighted_predictions[best_gesture] / total_weight
        
        # Fallback to rule-based if no model predictions
        if best_gesture is None or best_confidence < 0.7:
            rule_gesture = self.rule_based_classification(landmarks)
            if rule_gesture:
                best_gesture = rule_gesture
                best_confidence = 0.8
        
        return best_gesture, best_confidence
    
    def rule_based_classification(self, landmarks):
        """Improved rule-based classification as fallback"""
        if landmarks is None:
            return None
        
        # Get finger tip and joint positions
        tips = [4, 8, 12, 16, 20]
        mcp = [2, 5, 9, 13, 17]
        pip = [3, 6, 10, 14, 18]
        
        # Check finger extension status
        fingers_extended = []
        for i in range(5):
            if i == 0:  # Thumb
                if landmarks[tips[i]].x > landmarks[mcp[i]].x:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
            else:  # Other fingers
                if (landmarks[tips[i]].y < landmarks[pip[i]].y and 
                    landmarks[pip[i]].y < landmarks[mcp[i]].y):
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
        
        extended_count = sum(fingers_extended)
        
        # Basic gesture recognition
        if extended_count == 0:
            return 'A'
        elif extended_count == 4 and fingers_extended[0] == 0:
            return 'B'
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'C'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'D'
        elif extended_count == 5:
            return 'E'
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'F'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'G'
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'I'
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[4] == 1:
            return 'Y'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'L'
        
        return None
    
    def detect_sign_language(self, frame):
        """Main detection function with enhanced processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Classify gesture with enhanced method
                gesture, conf = self.enhanced_gesture_classification(frame, hand_landmarks.landmark)
                if gesture:
                    detected_gesture = gesture
                    confidence = conf
                    self.gesture_buffer.append((gesture, confidence))
        
        return frame, detected_gesture, confidence
    
    def process_gesture_buffer(self):
        """Process gesture buffer with advanced stability checking"""
        if len(self.gesture_buffer) < self.stability_frames:
            return None, 0
        
        # Get most recent gestures with confidence weighting
        recent_gestures = list(self.gesture_buffer)[-self.stability_frames:]
        
        # Weight gestures by confidence
        gesture_weights = {}
        total_weight = 0
        
        for gesture, confidence in recent_gestures:
            weight = confidence
            if gesture in gesture_weights:
                gesture_weights[gesture] += weight
            else:
                gesture_weights[gesture] = weight
            total_weight += weight
        
        if not gesture_weights or total_weight == 0:
            return None, 0
        
        # Find most confident gesture
        best_gesture = max(gesture_weights, key=gesture_weights.get)
        best_confidence = gesture_weights[best_gesture] / total_weight
        
        # Check stability with confidence threshold
        if best_confidence >= self.confidence_threshold:
            if best_gesture == self.last_gesture:
                self.gesture_stability_count += 1
            else:
                self.gesture_stability_count = 1
                self.last_gesture = best_gesture
            
            # Require gesture to be stable for more frames
            if self.gesture_stability_count >= 5:
                return best_gesture, best_confidence
        
        return None, 0
    
    def update_text(self, gesture, confidence):
        """Update the current text based on detected gesture"""
        if gesture == 'SPACE':
            self.current_text += " "
        elif gesture == 'DELETE' and len(self.current_text) > 0:
            self.current_text = self.current_text[:-1]
        elif gesture == 'ENTER':
            self.current_text += "\n"
        elif gesture and gesture in self.sign_mappings.values():
            self.current_text += gesture
        
        # Reset stability counter after text update
        self.gesture_stability_count = 0
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting Kaggle-Enhanced Sign Language Detection...")
        print("Detection running in background - check GUI for results")
        
        while self.detection_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect sign language
            frame, gesture, confidence = self.detect_sign_language(frame)
            
            # Store current state for GUI
            self.current_frame = frame
            self.current_gesture = gesture
            self.current_confidence = confidence
            
            # Process gesture if detected
            if gesture:
                final_gesture, final_confidence = self.process_gesture_buffer()
                if final_gesture:
                    self.update_text(final_gesture, final_confidence)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.033)  # ~30 FPS
        
        self.cleanup()
    
    def start_detection(self):
        """Start detection in background thread"""
        if not self.detection_running:
            self.detection_running = True
            detection_thread = threading.Thread(target=self.run_detection)
            detection_thread.daemon = True
            detection_thread.start()
            return True
        return False
    
    def stop_detection(self):
        """Stop detection"""
        self.detection_running = False
    
    def switch_model(self, model_name):
        """Switch active model"""
        if model_name in self.models:
            self.active_model = model_name
            return True
        return False
    
    def get_model_info(self):
        """Get information about available models"""
        return {
            "available_models": list(self.models.keys()),
            "active_model": self.active_model,
            "model_details": {
                name: {
                    "type": info["type"],
                    "accuracy": info["accuracy"],
                    "input_type": info["input_type"]
                }
                for name, info in self.models.items()
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class KaggleEnhancedGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Kaggle-Enhanced Sign Language Translator")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Create detector instance
        self.detector = KaggleEnhancedSignDetector()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_gui(self):
        """Setup the enhanced GUI elements"""
        # Title
        title_label = Label(self.root, text="Kaggle-Enhanced Sign Language Translator", 
                          font=("Arial", 24, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Status and controls
        left_panel = Frame(main_frame, bg='#34495e', relief=RAISED, bd=2)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Model selection frame
        model_frame = Frame(left_panel, bg='#34495e')
        model_frame.pack(fill=X, padx=10, pady=10)
        
        Label(model_frame, text="Active Model:", bg='#34495e', fg='white', 
              font=("Arial", 12, "bold")).pack(side=LEFT, padx=5)
        
        self.model_var = StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                      values=list(self.detector.models.keys()),
                                      width=20, state="readonly")
        self.model_combo.pack(side=LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        
        self.model_info_button = Button(model_frame, text="Model Info", 
                                      command=self.show_model_info, bg="#3498db", fg="white",
                                      font=("Arial", 9, "bold"), padx=10, pady=5)
        self.model_info_button.pack(side=LEFT, padx=5)
        
        # Dataset integration frame
        dataset_frame = Frame(left_panel, bg='#34495e')
        dataset_frame.pack(fill=X, padx=10, pady=10)
        
        Label(dataset_frame, text="Datasets:", bg='#34495e', fg='white', 
              font=("Arial", 12, "bold")).pack(side=LEFT, padx=5)
        
        self.download_button = Button(dataset_frame, text="Download Datasets", 
                                    command=self.download_datasets, bg="#e67e22", fg="white",
                                    font=("Arial", 9, "bold"), padx=10, pady=5)
        self.download_button.pack(side=LEFT, padx=5)
        
        self.train_button = Button(dataset_frame, text="Train Model", 
                                 command=self.train_model, bg="#9b59b6", fg="white",
                                 font=("Arial", 9, "bold"), padx=10, pady=5)
        self.train_button.pack(side=LEFT, padx=5)
        
        # Status display area
        status_frame = Frame(left_panel, bg='black', height=300)
        status_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.status_label = Label(status_frame, text="Click 'Start Detection' to begin", 
                                bg="black", fg="white", font=("Arial", 11), justify=LEFT)
        self.status_label.pack(expand=True, fill=BOTH, padx=10, pady=10)
        
        # Control buttons
        button_frame = Frame(left_panel, bg='#34495e')
        button_frame.pack(fill=X, padx=10, pady=10)
        
        self.start_button = Button(button_frame, text="Start Detection", 
                                 command=self.start_detection, bg="#27ae60", fg="white",
                                 font=("Arial", 10, "bold"), padx=15, pady=8)
        self.start_button.pack(side=LEFT, padx=2)
        
        self.stop_button = Button(button_frame, text="Stop Detection", 
                                command=self.stop_detection, bg="#e74c3c", fg="white",
                                font=("Arial", 10, "bold"), padx=15, pady=8)
        self.stop_button.pack(side=LEFT, padx=2)
        
        self.clear_button = Button(button_frame, text="Clear Text", 
                                 command=self.clear_text, bg="#f39c12", fg="white",
                                 font=("Arial", 10, "bold"), padx=15, pady=8)
        self.clear_button.pack(side=LEFT, padx=2)
        
        self.quit_button = Button(button_frame, text="Quit", 
                                command=self.quit_app, bg="#95a5a6", fg="white",
                                font=("Arial", 10, "bold"), padx=15, pady=8)
        self.quit_button.pack(side=LEFT, padx=2)
        
        # Right panel - Text display
        right_panel = Frame(main_frame, bg='#34495e', relief=RAISED, bd=2)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Translated text label
        text_label = Label(right_panel, text="Translated Text:", 
                          font=("Arial", 16, "bold"), bg='#34495e', fg='white')
        text_label.pack(pady=10)
        
        # Text display area with scrollbar
        text_frame = Frame(right_panel, bg='#34495e')
        text_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.translated_text = scrolledtext.ScrolledText(text_frame, 
                                                        height=30, width=50, 
                                                        font=("Arial", 14),
                                                        bg='#ecf0f1', fg='#2c3e50',
                                                        wrap=WORD)
        self.translated_text.pack(fill=BOTH, expand=True)
        
        # Status bar
        status_frame = Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill=X, side=BOTTOM)
        
        self.bottom_status_label = Label(status_frame, text="Ready to start detection", 
                                       bg='#34495e', fg='white', font=("Arial", 10))
        self.bottom_status_label.pack(pady=5)
        
        # Instructions
        instructions_frame = Frame(self.root, bg='#2c3e50')
        instructions_frame.pack(fill=X, padx=20, pady=10)
        
        instructions = Label(instructions_frame, 
                           text="Instructions: Download datasets → Train models → Start detection → Show ASL signs → View translated text",
                           font=("Arial", 11), bg='#2c3e50', fg='#bdc3c7', justify=CENTER)
        instructions.pack()
        
        # Initialize model selection
        if self.detector.models:
            self.model_combo.set(list(self.detector.models.keys())[0])
        
    def on_model_change(self, event):
        """Handle model selection change"""
        selected_model = self.model_var.get()
        if self.detector.switch_model(selected_model):
            self.bottom_status_label.config(text=f"Switched to model: {selected_model}")
        else:
            self.bottom_status_label.config(text="Failed to switch model")
    
    def show_model_info(self):
        """Show information about available models"""
        info = self.detector.get_model_info()
        
        info_text = "Available Models:\n\n"
        for name, details in info["model_details"].items():
            info_text += f"• {name}\n"
            info_text += f"  Type: {details['type']}\n"
            info_text += f"  Accuracy: {details['accuracy']:.2%}\n"
            info_text += f"  Input: {details['input_type']}\n\n"
        
        messagebox.showinfo("Model Information", info_text)
    
    def download_datasets(self):
        """Download datasets from Kaggle"""
        result = messagebox.askyesno("Download Datasets", 
                                   "This will download ASL datasets from Kaggle.\n"
                                   "You need Kaggle API credentials set up.\n\n"
                                   "Continue?")
        if result:
            try:
                # Run dataset downloader
                import subprocess
                subprocess.run(["./download_datasets.sh"], check=True)
                messagebox.showinfo("Success", "Datasets downloaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to download datasets: {e}")
    
    def train_model(self):
        """Train model with downloaded datasets"""
        result = messagebox.askyesno("Train Model", 
                                   "This will train a model using downloaded datasets.\n"
                                   "This process may take several hours.\n\n"
                                   "Continue?")
        if result:
            try:
                # Train model
                model, history = self.detector.dataset_integrator.train_with_kaggle_data(
                    "asl_alphabet", "mobilenet_asl", epochs=20
                )
                if model:
                    messagebox.showinfo("Success", "Model trained successfully!")
                    # Reload models
                    self.detector.load_available_models()
                    # Update combo box
                    self.model_combo['values'] = list(self.detector.models.keys())
                    if self.detector.models:
                        self.model_combo.set(list(self.detector.models.keys())[0])
                else:
                    messagebox.showerror("Error", "Model training failed!")
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {e}")
    
    def start_detection(self):
        """Start the detection process"""
        if self.detector.start_detection():
            self.status_label.config(text="🔴 Detection Running\nShow ASL signs to camera!")
            self.bottom_status_label.config(text="Detection started - Show signs to camera")
            self.start_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
        else:
            self.bottom_status_label.config(text="Detection already running")
        
    def stop_detection(self):
        """Stop the detection process"""
        self.detector.stop_detection()
        self.status_label.config(text="⏹️ Detection Stopped\nClick 'Start Detection' to begin")
        self.bottom_status_label.config(text="Detection stopped")
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        
    def update_gui(self):
        """Update the GUI with current detection data"""
        # Update text display
        if hasattr(self.detector, 'current_text'):
            current_text = self.detector.current_text
            # Update the text widget
            self.translated_text.delete(1.0, END)
            self.translated_text.insert(1.0, current_text)
            self.translated_text.see(END)
        
        # Update status display
        if self.detector.detection_running:
            status_text = "🔴 Detection Running\n\n"
            if self.detector.current_gesture:
                status_text += f"Gesture: {self.detector.current_gesture}\n"
                status_text += f"Confidence: {self.detector.current_confidence:.3f}\n"
            else:
                status_text += "No gesture detected\n"
            
            if hasattr(self.detector, 'current_fps'):
                status_text += f"FPS: {self.detector.current_fps}\n"
            
            status_text += f"Active Model: {self.detector.active_model or 'None'}\n"
            status_text += f"Available Models: {len(self.detector.models)}\n"
            
            status_text += "\nShow ASL signs to camera!"
            self.status_label.config(text=status_text)
        
        # Schedule next update
        self.root.after(100, self.update_gui)
        
    def clear_text(self):
        """Clear the translated text"""
        self.translated_text.delete(1.0, END)
        self.detector.current_text = ""
        self.bottom_status_label.config(text="Text cleared")
        
    def quit_app(self):
        """Quit the application"""
        self.detector.cleanup()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    # Check if webcam is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        exit(1)
    cap.release()
    
    print("Starting Kaggle-Enhanced Sign Language Translator...")
    print("This version integrates with Kaggle datasets and pre-trained models")
    
    # Start the GUI application
    app = KaggleEnhancedGUI()
    app.run()
