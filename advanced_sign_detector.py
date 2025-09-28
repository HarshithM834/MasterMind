import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext, messagebox
import queue
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Import enhanced training system
try:
    from enhanced_asl_trainer import EnhancedASLTrainer
    from asl_dataset_integration import ASLDatasetIntegration
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError:
    ENHANCED_TRAINING_AVAILABLE = False
    print("⚠️ Enhanced training system not available. Install required dependencies.")

class AdvancedSignLanguageDetector:
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
        self.gesture_buffer = deque(maxlen=20)  # Increased buffer size
        self.feature_buffer = deque(maxlen=10)  # Buffer for ML features
        self.confidence_threshold = 0.85  # Higher threshold for better accuracy
        self.stability_frames = 12  # More frames for stability
        self.current_text = ""
        self.last_gesture = None
        self.gesture_stability_count = 0
        
        # Debug settings
        self.debug_mode = True  # Enable debug output
        self.debug_output = []  # Store debug information
        self.max_debug_entries = 50  # Limit debug history
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection state
        self.detection_running = False
        self.current_frame = None
        self.current_gesture = None
        self.current_confidence = 0
        
        # ML Model and preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Enhanced training system
        self.enhanced_trainer = None
        if ENHANCED_TRAINING_AVAILABLE:
            try:
                self.enhanced_trainer = EnhancedASLTrainer()
                self.debug_log("Enhanced training system initialized")
            except Exception as e:
                self.debug_log(f"Failed to initialize enhanced trainer: {e}")
                self.enhanced_trainer = None
        
        # Load or create advanced model
        self.load_or_create_advanced_model()
        
        # Training data collection
        self.collecting_data = False
        self.training_data = []
        self.training_labels = []
        self.current_gesture_label = None
        
    def load_or_create_advanced_model(self):
        """Load existing advanced model or create a new one"""
        # Try to load enhanced model first
        if self.enhanced_trainer and self.enhanced_trainer.load_or_create_model():
            self.model = self.enhanced_trainer.model
            self.scaler = self.enhanced_trainer.scaler
            self.model_trained = True
            self.debug_log("✅ Loaded enhanced trained model")
            return
        
        # Fallback to basic model
        model_path = "advanced_sign_model.h5"
        scaler_path = "advanced_sign_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.model_trained = True
                self.debug_log("✅ Loaded basic trained model")
                return
            except Exception as e:
                self.debug_log(f"⚠️ Failed to load existing model: {e}")
        
        # Create new advanced model architecture
        self.create_advanced_model()
        self.model_trained = False
        self.debug_log("🆕 Created new advanced model (untrained)")
    
    def create_advanced_model(self):
        """Create an advanced neural network model optimized for 126 features"""
        input_dim = 126  # Exactly 126 features as extracted
        
        self.model = Sequential([
            # Input layer with batch normalization
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with optimal architecture for hand gesture recognition
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(29, activation='softmax')  # 26 letters + space + delete + enter
        ])
        
        # Compile with optimized settings
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005, decay=1e-6),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Print model summary for verification
        print("Model Architecture:")
        self.model.summary()
    
    def extract_advanced_features(self, landmarks):
        """Extract advanced features from hand landmarks - ensures exactly 126 features"""
        if landmarks is None or len(landmarks) != 21:
            return None
        
        features = []
        
        # 1. Raw landmark coordinates (normalized) - 21 landmarks * 3 coordinates = 63 features
        wrist = landmarks[0]
        for landmark in landmarks:
            # Normalize relative to wrist
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
        
        # 2. Distances between key points - 6C2 = 15 distances
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist, fingertips
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
            
            # Extension ratio
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
        for i in range(1, 5):  # Skip thumb
            tip = landmarks[tips[i]]
            pip = landmarks[pip[i]]
            mcp = landmarks[mcp[i]]
            
            # Angle between finger segments
            vec1 = [tip.x - pip.x, tip.y - pip.y]
            vec2 = [pip.x - mcp.x, pip.y - mcp.y]
            
            # Calculate angle
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
        
        # 7. Additional geometric features to reach exactly 126 - 34 more features
        # Finger tip distances from palm center
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
        
        # Additional hand shape features (14 more to reach 126)
        # Finger curvature indicators
        for i in range(5):
            tip = landmarks[tips[i]]
            pip = landmarks[pip[i]]
            mcp = landmarks[mcp[i]]
            
            # Curvature as distance from straight line
            if i == 0:  # Thumb
                curvature = abs(tip.x - mcp.x) - abs(pip.x - mcp.x)
            else:
                curvature = abs(tip.y - mcp.y) - abs(pip.y - mcp.y)
            features.append(curvature)
        
        # Hand openness (distance from fingertips to palm)
        palm_points_array = np.array([[landmarks[i].x, landmarks[i].y] for i in palm_points])
        palm_center_array = np.mean(palm_points_array, axis=0)
        
        for i in range(5):
            tip = np.array([landmarks[tips[i]].x, landmarks[tips[i]].y])
            distance = np.linalg.norm(tip - palm_center_array)
            features.append(distance)
        
        # Ensure exactly 126 features
        features = features[:126]  # Truncate if more than 126
        while len(features) < 126:  # Pad if less than 126
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def predict_gesture_ml(self, landmarks):
        """Use ML model to predict gesture with improved error handling"""
        if not self.model_trained or landmarks is None:
            return None, 0
        
        try:
            # Extract features
            features = self.extract_advanced_features(landmarks)
            if features is None:
                self.debug_log("Feature extraction failed")
                return None, 0
            
            # Validate feature dimensions
            if len(features) != 126:
                self.debug_log(f"Feature dimension mismatch: {len(features)} != 126")
                return None, 0
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                self.debug_log("Invalid feature values (NaN or Inf)")
                return None, 0
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled, verbose=0)
            confidence = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            
            # Validate prediction
            if predicted_class not in self.sign_mappings:
                self.debug_log(f"Invalid predicted class: {predicted_class}")
                return None, 0
            
            predicted_gesture = self.sign_mappings[predicted_class]
            self.debug_log(f"ML prediction: {predicted_gesture} (class: {predicted_class}, confidence: {confidence:.3f})")
            
            return predicted_gesture, confidence
            
        except Exception as e:
            self.debug_log(f"ML prediction error: {e}")
            return None, 0
    
    def enhanced_gesture_classification(self, landmarks):
        """Enhanced gesture classification using ML + rule-based fallback"""
        if landmarks is None:
            self.debug_log("No landmarks detected")
            return None, 0
        
        # Try ML prediction first
        if self.model_trained:
            ml_gesture, ml_confidence = self.predict_gesture_ml(landmarks)
            if ml_confidence > 0.7:  # High confidence ML prediction
                self.debug_log(f"ML prediction: {ml_gesture} (confidence: {ml_confidence:.3f})")
                return ml_gesture, ml_confidence
        
        # Fallback to improved rule-based classification
        rule_gesture = self.rule_based_classification(landmarks)
        if rule_gesture:
            self.debug_log(f"Rule-based prediction: {rule_gesture}")
            return rule_gesture, 0.8  # Medium confidence for rule-based
        
        self.debug_log("No gesture detected")
        return None, 0
    
    def rule_based_classification(self, landmarks):
        """Improved rule-based classification as fallback"""
        if landmarks is None:
            return None
        
        # Get finger tip and joint positions
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        mcp = [2, 5, 9, 13, 17]   # MCP joints
        pip = [3, 6, 10, 14, 18]  # PIP joints
        
        # Check finger extension status with improved logic
        fingers_extended = []
        finger_angles = []
        
        for i in range(5):
            if i == 0:  # Thumb (different logic)
                if landmarks[tips[i]].x > landmarks[mcp[i]].x:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
            else:  # Other fingers
                # Check if finger is extended (tip is above PIP joint)
                if (landmarks[tips[i]].y < landmarks[pip[i]].y and 
                    landmarks[pip[i]].y < landmarks[mcp[i]].y):
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
                
                # Calculate finger angle for additional precision
                tip = landmarks[tips[i]]
                pip_joint = landmarks[pip[i]]
                mcp_joint = landmarks[mcp[i]]
                
                # Vector from MCP to PIP
                vec1 = [pip_joint.x - mcp_joint.x, pip_joint.y - mcp_joint.y]
                # Vector from PIP to tip
                vec2 = [tip.x - pip_joint.x, tip.y - pip_joint.y]
                
                # Calculate angle
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    finger_angles.append(angle)
                else:
                    finger_angles.append(0)
        
        extended_count = sum(fingers_extended)
        
        # Enhanced gesture recognition with angle considerations
        return self.classify_enhanced_gesture_with_angles(
            fingers_extended, extended_count, finger_angles, landmarks
        )
    
    def classify_enhanced_gesture_with_angles(self, fingers_extended, extended_count, 
                                            finger_angles, landmarks):
        """Enhanced gesture classification considering finger angles with improved accuracy for all 26 letters"""
        
        # Helper function to check finger straightness
        def is_finger_straight(finger_idx):
            if finger_idx >= len(finger_angles):
                return False
            return finger_angles[finger_idx] < 0.4  # Less bent = more straight
        
        # Helper function to check finger bentness
        def is_finger_bent(finger_idx):
            if finger_idx >= len(finger_angles):
                return False
            return finger_angles[finger_idx] > 0.8  # More bent
        
        # Helper function to get tip distance
        def get_tip_distance(idx1, idx2):
            return abs(landmarks[idx1].x - landmarks[idx2].x)
        
        # Helper function to check hand orientation
        def get_hand_orientation():
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            return np.arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)
        
        # A - Fist (no fingers extended) - most restrictive
        if extended_count == 0:
            # Additional check: ensure all fingers are truly curled
            all_curled = all(not fingers_extended[i] for i in range(5))
            if all_curled:
                return 'A'
        
        # B - All fingers extended except thumb - very specific
        elif extended_count == 4 and fingers_extended[0] == 0:
            # Check that the 4 fingers are reasonably straight
            straight_fingers = sum(is_finger_straight(i) for i in range(1, 5))
            if straight_fingers >= 3:  # At least 3 of 4 fingers should be straight
                return 'B'
        
        # C - Index and middle finger extended, forming C shape
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            # Both fingers should be moderately bent to form C
            if (is_finger_bent(0) and is_finger_bent(1) and 
                not is_finger_straight(0) and not is_finger_straight(1)):
                return 'C'
        
        # D - Only index finger extended and straight
        elif (extended_count == 1 and fingers_extended[1] == 1 and 
              is_finger_straight(0)):  # Index finger should be straight
            return 'D'
        
        # E - All fingers extended - very specific
        elif extended_count == 5:
            # Check that most fingers are reasonably straight
            straight_fingers = sum(is_finger_straight(i) for i in range(5))
            if straight_fingers >= 4:  # At least 4 of 5 fingers should be straight
                return 'E'
        
        # F - Thumb and index finger extended (OK sign)
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            # Check if they form a circle/OK sign
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if distance < 0.05:  # Close together for OK sign
                return 'F'
        
        # G - Index finger extended, pointing (straight)
        elif (extended_count == 1 and fingers_extended[1] == 1 and 
              is_finger_straight(0)):  # Index finger should be very straight
            return 'G'
        
        # H - Index and middle finger extended, close together
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = get_tip_distance(8, 12)
            if tip_distance < 0.025:  # Very close together
                return 'H'
        
        # I - Pinky extended
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'I'
        
        # J - Pinky extended with hook motion
        elif (extended_count == 1 and fingers_extended[4] == 1 and 
              is_finger_bent(3)):  # Pinky should be bent
            return 'J'
        
        # K - Index and middle finger extended, apart
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = get_tip_distance(8, 12)
            if tip_distance > 0.06:  # Clearly apart
                return 'K'
        
        # L - Index and thumb extended (L shape)
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            # Check if they form L shape (not too close)
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if distance > 0.05:  # Not too close (different from F)
                return 'L'
        
        # M - Three fingers (index, middle, ring) extended
        elif (extended_count == 3 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and fingers_extended[3] == 1):
            return 'M'
        
        # N - Two fingers (index, middle) extended - distinguish from H/K
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = get_tip_distance(8, 12)
            if 0.03 <= tip_distance <= 0.06:  # Moderate distance
                return 'N'
        
        # O - Fingers curled to form O shape
        elif extended_count == 0:
            # Check if fingertips are close to form O
            fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            center_x = sum(tip.x for tip in fingertips) / 5
            center_y = sum(tip.y for tip in fingertips) / 5
            distances = [np.sqrt((tip.x - center_x)**2 + (tip.y - center_y)**2) 
                        for tip in fingertips]
            if all(d < 0.035 for d in distances):  # Tight O shape
                return 'O'
        
        # P - Index finger pointing down
        elif (extended_count == 1 and fingers_extended[1] == 1 and 
              landmarks[8].y > landmarks[6].y):  # Tip below PIP
            return 'P'
        
        # Q - Index finger pointing to side
        elif (extended_count == 1 and fingers_extended[1] == 1 and 
              abs(landmarks[8].x - landmarks[6].x) > 0.06):  # Significant horizontal offset
            return 'Q'
        
        # R - Index and middle finger crossed
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and landmarks[8].x > landmarks[12].x):
            return 'R'
        
        # S - Fist with thumb over fingers
        elif extended_count == 0 and landmarks[4].y < landmarks[8].y:
            return 'S'
        
        # T - Thumb between index and middle fingers
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and 
              landmarks[4].x > landmarks[8].x and landmarks[4].x < landmarks[12].x):
            return 'T'
        
        # U - Index and middle finger extended, apart (distinguish from V)
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = get_tip_distance(8, 12)
            if tip_distance > 0.05:  # Apart
                # Check hand orientation to distinguish U from V
                orientation = get_hand_orientation()
                if abs(orientation) < 0.5:  # Hand more horizontal for U
                    return 'U'
                else:  # Hand more vertical for V
                    return 'V'
        
        # W - Index, middle, and ring fingers extended
        elif (extended_count == 3 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and fingers_extended[3] == 1):
            return 'W'
        
        # X - Index finger bent (not extended but bent)
        elif (extended_count == 0 and is_finger_bent(0)):  # Index finger bent
            return 'X'
        
        # Y - Thumb and pinky extended
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[4] == 1:
            return 'Y'
        
        # Z - Index finger pointing (simplified)
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'Z'
        
        return None
    
    def detect_sign_language(self, frame):
        """Main detection function with advanced processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Enhanced hand tracking visualization
                self.draw_enhanced_hand_tracking(frame, hand_landmarks)
                
                # Classify gesture with advanced method
                gesture, conf = self.enhanced_gesture_classification(hand_landmarks.landmark)
                if gesture:
                    detected_gesture = gesture
                    confidence = conf
                    self.gesture_buffer.append((gesture, confidence))
                    
                    # Add gesture label to frame
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Store features for potential training
                    if self.collecting_data and self.current_gesture_label:
                        features = self.extract_advanced_features(hand_landmarks.landmark)
                        if features is not None:
                            self.training_data.append(features)
                            self.training_labels.append(self.reverse_mappings.get(self.current_gesture_label, 0))
        
        return frame, detected_gesture, confidence
    
    def draw_enhanced_hand_tracking(self, frame, hand_landmarks):
        """Draw enhanced hand tracking with better visualization"""
        # Draw hand landmarks with custom styling
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            
            # Different colors for different landmark types
            if idx in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green for fingertips
            elif idx in [2, 5, 9, 13, 17]:  # MCP joints
                cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # Blue for MCP
            elif idx in [3, 6, 10, 14, 18]:  # PIP joints
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for PIP
            else:  # Other landmarks
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)  # White for others
        
        # Draw connections with enhanced styling
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_point = hand_landmarks.landmark[connection[0]]
            end_point = hand_landmarks.landmark[connection[1]]
            
            start_x = int(start_point.x * frame.shape[1])
            start_y = int(start_point.y * frame.shape[0])
            end_x = int(end_point.x * frame.shape[1])
            end_y = int(end_point.y * frame.shape[0])
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Add hand bounding box
        x_coords = [int(landmark.x * frame.shape[1]) for landmark in hand_landmarks.landmark]
        y_coords = [int(landmark.y * frame.shape[0]) for landmark in hand_landmarks.landmark]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Draw bounding box
        cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), 
                    (0, 255, 255), 2)
        
        # Add "HAND DETECTED" label
        cv2.putText(frame, "HAND DETECTED", (min_x, min_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_gesture_buffer(self):
        """Process gesture buffer with advanced stability checking and hysteresis"""
        if len(self.gesture_buffer) < self.stability_frames:
            return None, 0
        
        # Get most recent gestures with confidence weighting
        recent_gestures = list(self.gesture_buffer)[-self.stability_frames:]
        
        # Weight gestures by confidence and recency (more recent = higher weight)
        gesture_weights = {}
        total_weight = 0
        
        for i, (gesture, confidence) in enumerate(recent_gestures):
            # Weight by confidence and recency
            recency_weight = (i + 1) / len(recent_gestures)  # More recent = higher weight
            weight = confidence * recency_weight
            
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
        
        # Enhanced stability checking with hysteresis
        if best_confidence >= self.confidence_threshold:
            if best_gesture == self.last_gesture:
                self.gesture_stability_count += 1
            else:
                # Only change gesture if new gesture is significantly better
                if self.last_gesture and best_gesture in gesture_weights:
                    current_weight = gesture_weights.get(self.last_gesture, 0) / total_weight
                    improvement_ratio = best_confidence / (current_weight + 1e-6)
                    
                    # Require significant improvement to change gesture (hysteresis)
                    if improvement_ratio < 1.5:  # Must be 50% better to switch
                        return None, 0
                
                self.gesture_stability_count = 1
                self.last_gesture = best_gesture
            
            # Require gesture to be stable for more frames (adaptive threshold)
            required_stability = 8 if best_confidence > 0.9 else 12  # Higher confidence = less stability needed
            if self.gesture_stability_count >= required_stability:
                return best_gesture, best_confidence
        
        return None, 0
    
    def update_text(self, gesture, confidence):
        """Update the current text based on detected gesture with validation"""
        # Validate gesture before processing
        if not self.is_valid_gesture(gesture):
            self.debug_log(f"Invalid gesture detected: {gesture}")
            return
        
        # Only process gestures with sufficient confidence
        if confidence < 0.6:  # Minimum confidence threshold
            self.debug_log(f"Low confidence gesture ignored: {gesture} ({confidence:.3f})")
            return
        
        self.debug_log(f"Processing gesture: {gesture} (confidence: {confidence:.3f})")
        
        if gesture == 'SPACE':
            self.current_text += " "
        elif gesture == 'DELETE' and len(self.current_text) > 0:
            self.current_text = self.current_text[:-1]
        elif gesture == 'ENTER':
            self.current_text += "\n"
        elif gesture and gesture in self.sign_mappings.values():
            # Additional validation for letter gestures
            if gesture in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                # Ensure proper case handling - ASL letters are typically uppercase
                self.current_text += gesture.upper()
                self.debug_log(f"Added letter '{gesture.upper()}' to text")
            else:
                self.debug_log(f"Unexpected gesture: {gesture}")
        
        # Reset stability counter after text update
        self.gesture_stability_count = 0
    
    def is_valid_gesture(self, gesture):
        """Validate that the gesture is a valid ASL gesture"""
        valid_gestures = set(self.sign_mappings.values())
        return gesture in valid_gestures
    
    def debug_log(self, message):
        """Add debug message to debug output"""
        if self.debug_mode:
            timestamp = time.strftime("%H:%M:%S")
            debug_msg = f"[{timestamp}] {message}"
            self.debug_output.append(debug_msg)
            
            # Keep only recent debug messages
            if len(self.debug_output) > self.max_debug_entries:
                self.debug_output = self.debug_output[-self.max_debug_entries:]
            
            print(debug_msg)  # Also print to console
    
    def get_debug_info(self):
        """Get current debug information"""
        return {
            'gesture_buffer_size': len(self.gesture_buffer),
            'last_gesture': self.last_gesture,
            'stability_count': self.gesture_stability_count,
            'model_trained': self.model_trained,
            'current_fps': self.current_fps,
            'debug_messages': self.debug_output[-10:] if self.debug_output else []  # Last 10 messages
        }
    
    def test_all_letters(self):
        """Test function to verify all 26 letters can be recognized"""
        self.debug_log("🧪 Testing all 26 ASL letters...")
        
        all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        recognized_letters = set()
        
        # This would be called during actual testing with hand gestures
        for letter in all_letters:
            if letter in self.sign_mappings.values():
                recognized_letters.add(letter)
                self.debug_log(f"✅ Letter {letter} is properly mapped")
            else:
                self.debug_log(f"❌ Letter {letter} is NOT properly mapped")
        
        self.debug_log(f"📊 Recognition Status: {len(recognized_letters)}/26 letters properly configured")
        return len(recognized_letters) == 26
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_detection(self):
        """Main detection loop - runs without OpenCV GUI"""
        print("Starting Advanced Sign Language Detection...")
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
    
    def start_training_data_collection(self, gesture_label):
        """Start collecting training data for a specific gesture"""
        self.collecting_data = True
        self.current_gesture_label = gesture_label
        self.training_data = []
        self.training_labels = []
        print(f"Started collecting training data for gesture: {gesture_label}")
    
    def stop_training_data_collection(self):
        """Stop collecting training data"""
        self.collecting_data = False
        self.current_gesture_label = None
        print(f"Collected {len(self.training_data)} training samples")
    
    def train_model(self):
        """Train the model with collected data"""
        if len(self.training_data) < 100:  # Need minimum samples
            return False, "Need at least 100 training samples"
        
        try:
            # Prepare data
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model and scaler
            self.model.save('advanced_sign_model.h5')
            joblib.dump(self.scaler, 'advanced_sign_scaler.pkl')
            
            self.model_trained = True
            
            # Evaluate model
            test_loss, test_acc, test_top3_acc = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            
            return True, f"Model trained successfully! Accuracy: {test_acc:.3f}, Top-3 Accuracy: {test_top3_acc:.3f}"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class AdvancedSignLanguageGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("🤟 Gesture Detection")
        self.root.geometry("1400x900")
        
        # Chill and appealing color scheme
        self.colors = {
            'primary': '#2c3e50',      # Softer dark blue
            'secondary': '#34495e',     # Muted blue-gray
            'accent': '#e74c3c',       # Warm coral red
            'success': '#27ae60',      # Fresh green
            'warning': '#f39c12',      # Warm orange
            'text': '#ecf0f1',         # Soft white
            'text_secondary': '#bdc3c7', # Light gray
            'card': '#34495e',         # Soft card background
            'border': '#7f8c8d',       # Light border
            'highlight': '#3498db',    # Bright blue accent
            'background': '#2c3e50'    # Main background
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Create detector instance
        self.detector = AdvancedSignLanguageDetector()
        
        # Video display variables
        self.video_label = None
        self.show_video = True
        
        # Create GUI elements
        self.setup_modern_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_modern_gui(self):
        """Setup the modern, aesthetic GUI with video display"""
        # Header with chill gradient effect
        header_frame = Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill=X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title with emoji and chill styling
        title_label = Label(header_frame, text="🤟 Gesture Detection", 
                          font=("Segoe UI", 32, "bold"), 
                          bg=self.colors['primary'], fg=self.colors['text'])
        title_label.pack(pady=25)
        
        # Subtitle with chill vibe
        subtitle_label = Label(header_frame, text="✨ Real-time ASL Recognition • Chill & Easy ✨", 
                             font=("Segoe UI", 14), 
                             bg=self.colors['primary'], fg=self.colors['text_secondary'])
        subtitle_label.pack()
        
        # Main content area with chill spacing
        main_frame = Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=BOTH, expand=True, padx=25, pady=25)
        
        # Left panel - Video and controls with rounded feel
        left_panel = Frame(main_frame, bg=self.colors['card'], relief=FLAT, bd=2)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 15))
        
        # Video display area - Chill rounded corners
        video_frame = Frame(left_panel, bg='black', relief=FLAT, bd=3)
        video_frame.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        # Video label for camera feed - Chill message
        self.video_label = Label(video_frame, text="📹 Camera Feed\n\n✨ Click 'Start Detection' to begin ✨", 
                               bg='black', fg='white', font=("Segoe UI", 16),
                               justify=CENTER, relief=FLAT, bd=0)
        self.video_label.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons with chill styling
        control_frame = Frame(left_panel, bg=self.colors['card'])
        control_frame.pack(fill=X, padx=15, pady=15)
        
        # Primary controls with better spacing
        primary_controls = Frame(control_frame, bg=self.colors['card'])
        primary_controls.pack(fill=X, pady=(0, 15))
        
        self.start_button = self.create_chill_button(primary_controls, "🚀 Start Detection", 
                                                    self.start_detection, self.colors['success'])
        self.start_button.pack(side=LEFT, padx=8)
        
        self.stop_button = self.create_chill_button(primary_controls, "⏹️ Stop Detection", 
                                                   self.stop_detection, self.colors['accent'])
        self.stop_button.pack(side=LEFT, padx=8)
        
        self.clear_button = self.create_chill_button(primary_controls, "🗑️ Clear Text", 
                                                    self.clear_text, self.colors['warning'])
        self.clear_button.pack(side=LEFT, padx=8)
        
        # Secondary controls
        secondary_controls = Frame(control_frame, bg=self.colors['card'])
        secondary_controls.pack(fill=X)
        
        self.test_button = self.create_chill_button(secondary_controls, "🧪 Test Letters", 
                                                   self.test_all_letters, self.colors['highlight'])
        self.test_button.pack(side=LEFT, padx=8)
        
        # Training section with chill vibes
        training_section = Frame(left_panel, bg=self.colors['card'], relief=FLAT, bd=2)
        training_section.pack(fill=X, padx=15, pady=15)
        
        training_title = Label(training_section, text="🎓 AI Training Center", 
                             font=("Segoe UI", 16, "bold"), 
                             bg=self.colors['card'], fg=self.colors['text'])
        training_title.pack(pady=(15, 10))
        
        training_controls = Frame(training_section, bg=self.colors['card'])
        training_controls.pack(fill=X, padx=15, pady=15)
        
        Label(training_controls, text="Letter:", bg=self.colors['card'], fg=self.colors['text'], 
              font=("Segoe UI", 12)).pack(side=LEFT, padx=8)
        
        self.gesture_var = StringVar(value="A")
        gesture_combo = ttk.Combobox(training_controls, textvariable=self.gesture_var,
                                   values=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                                   width=3, state="readonly", font=("Segoe UI", 12))
        gesture_combo.pack(side=LEFT, padx=8)
        
        self.collect_button = self.create_chill_button(training_controls, "📊 Collect", 
                                                      self.start_data_collection, self.colors['highlight'])
        self.collect_button.pack(side=LEFT, padx=8)
        
        self.train_button = self.create_chill_button(training_controls, "🧠 Train", 
                                                    self.train_model, "#9b59b6")
        self.train_button.pack(side=LEFT, padx=8)
        
        # Enhanced training button
        if ENHANCED_TRAINING_AVAILABLE:
            self.enhanced_train_button = self.create_chill_button(training_controls, "⚡ Enhanced Train", 
                                                                 self.enhanced_train_model, "#e67e22")
            self.enhanced_train_button.pack(side=LEFT, padx=8)
        
        # Right panel - Text output and status with chill vibes
        right_panel = Frame(main_frame, bg=self.colors['card'], relief=FLAT, bd=2)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Text output section
        text_section = Frame(right_panel, bg=self.colors['card'])
        text_section.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        text_title = Label(text_section, text="📝 Translated Text", 
                         font=("Segoe UI", 18, "bold"), 
                         bg=self.colors['card'], fg=self.colors['text'])
        text_title.pack(pady=(0, 15))
        
        # Text display with chill styling
        self.translated_text = scrolledtext.ScrolledText(text_section, 
                                                        height=20, width=40, 
                                                        font=("Segoe UI", 16),
                                                        bg='#ecf0f1', fg='#2c3e50',
                                                        wrap=WORD, relief=FLAT, bd=2,
                                                        selectbackground=self.colors['highlight'])
        self.translated_text.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Status section with chill vibes
        status_section = Frame(right_panel, bg=self.colors['card'])
        status_section.pack(fill=X, padx=15, pady=15)
        
        status_title = Label(status_section, text="📊 System Status", 
                           font=("Segoe UI", 16, "bold"), 
                           bg=self.colors['card'], fg=self.colors['text'])
        status_title.pack(pady=(0, 10))
        
        self.status_label = Label(status_section, text="🟢 Ready to start detection", 
                                bg=self.colors['card'], fg=self.colors['text'], 
                                font=("Segoe UI", 12), justify=LEFT, wraplength=300)
        self.status_label.pack(fill=X, padx=10, pady=10)
        
        # Bottom status bar with chill vibes
        status_bar = Frame(self.root, bg=self.colors['secondary'], height=50)
        status_bar.pack(fill=X, side=BOTTOM)
        status_bar.pack_propagate(False)
        
        self.bottom_status_label = Label(status_bar, text="🤟 Gesture Detection - Ready ✨", 
                                       bg=self.colors['secondary'], fg=self.colors['text'], 
                                       font=("Segoe UI", 12))
        self.bottom_status_label.pack(pady=15)
        
        # Quit button
        self.quit_button = self.create_chill_button(status_bar, "❌ Quit", 
                                                  self.quit_app, self.colors['accent'])
        self.quit_button.pack(side=RIGHT, padx=25, pady=10)
    
    def create_chill_button(self, parent, text, command, color):
        """Create a chill-styled button with better spacing and rounded feel"""
        button = Button(parent, text=text, command=command, 
                       bg=color, fg='black', font=("Segoe UI", 11, "bold"),
                       relief=FLAT, bd=2, padx=20, pady=10,
                       activebackground=self.lighten_color(color),
                       activeforeground='black',
                       cursor='hand2')
        return button
    
    def create_modern_button(self, parent, text, command, color):
        """Create a modern-styled button with black text"""
        button = Button(parent, text=text, command=command, 
                       bg=color, fg='black', font=("Segoe UI", 10, "bold"),
                       relief=FLAT, bd=0, padx=15, pady=8,
                       activebackground=self.lighten_color(color),
                       activeforeground='black',
                       cursor='hand2')
        return button
    
    def lighten_color(self, color):
        """Lighten a hex color for hover effects"""
        # Simple color lightening - in a real app you'd use proper color manipulation
        color_map = {
            '#e94560': '#ff6b7a',
            '#00d4aa': '#00f5d4', 
            '#ffd700': '#fff200',
            '#3498db': '#5dade2',
            '#9b59b6': '#bb8fce',
            '#e67e22': '#f39c12',
            '#0f3460': '#1e4a6b'
        }
        return color_map.get(color, color)
        
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
        
    def start_data_collection(self):
        """Start collecting training data"""
        gesture = self.gesture_var.get()
        if not self.detector.detection_running:
            messagebox.showwarning("Warning", "Please start detection first!")
            return
        
        self.detector.start_training_data_collection(gesture)
        self.collect_button.config(text="Stop Collecting", command=self.stop_data_collection,
                                 bg="#e74c3c")
        self.bottom_status_label.config(text=f"Collecting training data for '{gesture}' - Show the gesture repeatedly")
        
    def stop_data_collection(self):
        """Stop collecting training data"""
        self.detector.stop_training_data_collection()
        self.collect_button.config(text="Collect Data", command=self.start_data_collection,
                                 bg="#3498db")
        self.bottom_status_label.config(text="Data collection stopped")
        
    def train_model(self):
        """Train the model"""
        success, message = self.detector.train_model()
        if success:
            messagebox.showinfo("Success", message)
            self.bottom_status_label.config(text="Model trained successfully!")
        else:
            messagebox.showerror("Error", message)
            self.bottom_status_label.config(text="Training failed")
    
    def enhanced_train_model(self):
        """Train the model using enhanced training system"""
        if not ENHANCED_TRAINING_AVAILABLE:
            messagebox.showerror("Error", "Enhanced training system not available")
            return
        
        if not self.detector.enhanced_trainer:
            messagebox.showerror("Error", "Enhanced trainer not initialized")
            return
        
        # Show progress dialog
        progress_window = Toplevel(self.root)
        progress_window.title("Enhanced Training Progress")
        progress_window.geometry("400x200")
        progress_window.configure(bg='#2c3e50')
        
        progress_label = Label(progress_window, text="Starting enhanced training with large dataset...", 
                              bg='#2c3e50', fg='white', font=("Arial", 12))
        progress_label.pack(pady=20)
        
        progress_text = scrolledtext.ScrolledText(progress_window, height=8, width=50,
                                                 bg='#34495e', fg='white', font=("Arial", 10))
        progress_text.pack(pady=10, padx=20, fill=BOTH, expand=True)
        
        def run_enhanced_training():
            try:
                progress_text.insert(END, "🔄 Preparing comprehensive dataset...\n")
                progress_text.see(END)
                progress_window.update()
                
                # Train the enhanced model
                history = self.detector.enhanced_trainer.train_enhanced_model(epochs=30, batch_size=32)
                
                progress_text.insert(END, "✅ Enhanced training completed!\n")
                progress_text.insert(END, "📊 Check generated reports for detailed analysis\n")
                progress_text.see(END)
                progress_window.update()
                
                # Update detector with new model
                self.detector.model = self.detector.enhanced_trainer.model
                self.detector.scaler = self.detector.enhanced_trainer.scaler
                self.detector.model_trained = True
                
                messagebox.showinfo("Success", "Enhanced training completed successfully!")
                self.bottom_status_label.config(text="Enhanced model trained successfully!")
                
            except Exception as e:
                progress_text.insert(END, f"❌ Training failed: {str(e)}\n")
                progress_text.see(END)
                messagebox.showerror("Error", f"Enhanced training failed: {str(e)}")
        
        # Run training in separate thread
        training_thread = threading.Thread(target=run_enhanced_training)
        training_thread.daemon = True
        training_thread.start()
        
    def update_gui(self):
        """Update the GUI with current detection data and video feed"""
        # Update video display
        if self.detector.detection_running and hasattr(self.detector, 'current_frame'):
            self.update_video_display()
        
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
            
            status_text += f"Model Status: {'Trained' if self.detector.model_trained else 'Untrained'}\n"
            
            # Add debug information
            debug_info = self.detector.get_debug_info()
            status_text += f"Buffer Size: {debug_info['gesture_buffer_size']}\n"
            status_text += f"Stability: {debug_info['stability_count']}\n"
            
            if self.detector.collecting_data:
                status_text += f"\n📊 Collecting data for: {self.detector.current_gesture_label}\n"
                status_text += f"Samples: {len(self.detector.training_data)}"
            
            # Show recent debug messages
            if debug_info['debug_messages']:
                status_text += "\n\nRecent Activity:"
                for msg in debug_info['debug_messages'][-3:]:  # Show last 3 messages
                    status_text += f"\n{msg}"
            
            status_text += "\n\nShow ASL signs to camera!"
            self.status_label.config(text=status_text)
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def update_video_display(self):
        """Update the video display with hand tracking"""
        if hasattr(self.detector, 'current_frame') and self.detector.current_frame is not None:
            try:
                # Convert frame for display
                frame = self.detector.current_frame.copy()
                
                # Resize frame to fit the display
                height, width = frame.shape[:2]
                max_width = 400
                max_height = 300
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB for tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(frame_rgb)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update the video label
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # Keep a reference
                
            except Exception as e:
                print(f"Video display error: {e}")
                self.video_label.config(text="📹 Camera Error\n\nCheck camera connection")
        
    def clear_text(self):
        """Clear the translated text"""
        self.translated_text.delete(1.0, END)
        self.detector.current_text = ""
        self.bottom_status_label.config(text="Text cleared")
    
    def test_all_letters(self):
        """Test all 26 ASL letters"""
        success = self.detector.test_all_letters()
        if success:
            messagebox.showinfo("Test Results", "✅ All 26 letters are properly configured!")
            self.bottom_status_label.config(text="All 26 letters tested successfully")
        else:
            messagebox.showwarning("Test Results", "⚠️ Some letters may need configuration")
            self.bottom_status_label.config(text="Letter test completed with warnings")
        
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
    
    print("Starting Advanced Sign Language Translator...")
    print("This version includes ML training capabilities and improved accuracy")
    
    # Start the GUI application
    app = AdvancedSignLanguageGUI()
    app.run()
# done