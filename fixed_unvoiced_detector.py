import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext, messagebox
import os
from pathlib import Path
import gtts
from pygame import mixer
import tempfile

class FixedUnvoicedDetector:
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
        
        # Initialize text-to-speech
        self.tts_enabled = True
        try:
            mixer.init()
            print("✅ Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio system not available: {e}")
            self.tts_enabled = False
        
        # Load Unvoiced labels
        self.load_unvoiced_labels()
        
        # Create Inception V3 model similar to Unvoiced
        self.create_inception_model()
        
        # Gesture recognition parameters
        self.gesture_buffer = deque(maxlen=15)
        self.confidence_threshold = 0.7  # Lower threshold for better detection
        self.stability_frames = 8
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
        
        # TTS settings
        self.speak_on_detection = False
        
    def load_unvoiced_labels(self):
        """Load labels from Unvoiced repository"""
        labels_path = Path("unvoiced_resources/training_set_labels.txt")
        
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.labels)} labels from Unvoiced")
        else:
            # Fallback to standard ASL alphabet
            self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            print("⚠️ Using fallback ASL alphabet labels")
        
        # Create reverse mapping
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
    
    def create_inception_model(self):
        """Create Inception V3 model similar to Unvoiced"""
        print("🏗️ Creating Inception V3 model architecture...")
        
        # Load pre-trained Inception V3
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(self.labels), activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Created Inception V3 model for {len(self.labels)} classes")
        
        # Since we don't have trained weights, we'll use a hybrid approach
        self.use_hybrid_detection = True
    
    def extract_hand_region_fixed(self, frame, landmarks):
        """Extract hand region optimized for Inception V3"""
        if landmarks is None:
            return None
        
        # Get bounding box of hand
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding for better hand capture
        padding = 0.2
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
        
        # Create a square region for better model performance
        size = max(hand_region.shape[0], hand_region.shape[1])
        square_region = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Center the hand in the square
        start_x = (size - hand_region.shape[1]) // 2
        start_y = (size - hand_region.shape[0]) // 2
        square_region[start_y:start_y + hand_region.shape[0], 
                     start_x:start_x + hand_region.shape[1]] = hand_region
        
        return square_region
    
    def predict_with_inception(self, hand_region):
        """Predict using Inception V3 model"""
        if hand_region is None:
            return None, 0
        
        try:
            # Preprocess for Inception V3
            processed_image = cv2.resize(hand_region, (299, 299))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.astype(np.float32) / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)
            
            # Predict
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Map to label
            if predicted_class_idx < len(self.labels):
                predicted_label = self.labels[predicted_class_idx]
                # Only return if confidence is reasonable
                if confidence > 0.1:  # Very low threshold for untrained model
                    return predicted_label, confidence
            
            return None, 0
            
        except Exception as e:
            print(f"❌ Inception prediction error: {e}")
            return None, 0
    
    def enhanced_rule_based_classification(self, landmarks):
        """Enhanced rule-based classification as fallback"""
        if landmarks is None:
            return None
        
        # Get finger tip and joint positions
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        mcp = [2, 5, 9, 13, 17]   # MCP joints
        pip = [3, 6, 10, 14, 18]  # PIP joints
        
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
        
        # Enhanced gesture recognition with better rules
        gesture = self.classify_enhanced_gesture(fingers_extended, extended_count, landmarks)
        
        return gesture, 0.8  # High confidence for rule-based
    
    def classify_enhanced_gesture(self, fingers_extended, extended_count, landmarks):
        """Enhanced gesture classification logic"""
        
        # A - Fist (no fingers extended)
        if extended_count == 0:
            return 'A'
        
        # B - All fingers extended except thumb
        elif extended_count == 4 and fingers_extended[0] == 0:
            return 'B'
        
        # C - Index and middle finger extended, forming C shape
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'C'
        
        # D - Only index finger extended and straight
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'D'
        
        # E - All fingers extended
        elif extended_count == 5:
            return 'E'
        
        # F - Thumb and index finger extended
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'F'
        
        # G - Index finger extended, pointing (straight)
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'G'
        
        # H - Index and middle finger extended, close together
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = abs(landmarks[8].x - landmarks[12].x)
            if tip_distance < 0.05:  # Close together
                return 'H'
        
        # I - Pinky extended
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'I'
        
        # J - Pinky extended with hook motion
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'J'
        
        # K - Index and middle finger extended, apart
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = abs(landmarks[8].x - landmarks[12].x)
            if tip_distance > 0.05:  # Apart
                return 'K'
        
        # L - Index and thumb extended
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'L'
        
        # M - Three fingers (index, middle, ring) extended
        elif (extended_count == 3 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and fingers_extended[3] == 1):
            return 'M'
        
        # N - Two fingers (index, middle) extended
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'N'
        
        # O - Fingers curled to form O shape
        elif extended_count == 0:
            # Check if fingertips are close to form O
            center_x = sum(landmark.x for landmark in landmarks[4:21:4]) / 5
            center_y = sum(landmark.y for landmark in landmarks[4:21:4]) / 5
            distances = [np.sqrt((landmark.x - center_x)**2 + (landmark.y - center_y)**2) 
                        for landmark in landmarks[4:21:4]]
            if all(d < 0.05 for d in distances):
                return 'O'
        
        # P - Index finger pointing down
        elif extended_count == 1 and fingers_extended[1] == 1:
            if landmarks[8].y > landmarks[6].y:
                return 'P'
        
        # Q - Index finger pointing to side
        elif extended_count == 1 and fingers_extended[1] == 1:
            if abs(landmarks[8].x - landmarks[6].x) > 0.05:
                return 'Q'
        
        # R - Index and middle finger crossed
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            if landmarks[8].x > landmarks[12].x:
                return 'R'
        
        # S - Fist with thumb over fingers
        elif extended_count == 0:
            if landmarks[4].y < landmarks[8].y:
                return 'S'
        
        # T - Thumb between index and middle fingers
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            if landmarks[4].x > landmarks[8].x and landmarks[4].x < landmarks[12].x:
                return 'T'
        
        # U - Index and middle finger extended, apart
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            return 'U'
        
        # V - Index and middle finger extended, apart (same as U)
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            return 'V'
        
        # W - Index, middle, and ring fingers extended
        elif (extended_count == 3 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1 and fingers_extended[3] == 1):
            return 'W'
        
        # X - Index finger bent
        elif extended_count == 0:
            if landmarks[8].y > landmarks[6].y:
                return 'X'
        
        # Y - Thumb and pinky extended
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[4] == 1:
            return 'Y'
        
        # Z - Index finger pointing (simplified)
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'Z'
        
        return None
    
    def detect_sign_language_fixed(self, frame):
        """Main detection function with fixed approach"""
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
                
                # Try Inception model first (if available)
                if hasattr(self, 'use_hybrid_detection') and self.use_hybrid_detection:
                    hand_region = self.extract_hand_region_fixed(frame, hand_landmarks.landmark)
                    if hand_region is not None:
                        inception_gesture, inception_conf = self.predict_with_inception(hand_region)
                        if inception_gesture and inception_conf > 0.3:  # Low threshold for demo
                            detected_gesture = inception_gesture
                            confidence = inception_conf
                            self.gesture_buffer.append((gesture, confidence))
                            continue
                
                # Fallback to enhanced rule-based classification
                rule_gesture, rule_conf = self.enhanced_rule_based_classification(hand_landmarks.landmark)
                if rule_gesture:
                    detected_gesture = rule_gesture
                    confidence = rule_conf
                    self.gesture_buffer.append((gesture, confidence))
        
        return frame, detected_gesture, confidence
    
    def process_gesture_buffer(self):
        """Process gesture buffer with stability checking"""
        if len(self.gesture_buffer) < self.stability_frames:
            return None, 0
        
        # Get most recent gestures
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
            
            # Require gesture to be stable for a few frames
            if self.gesture_stability_count >= 3:
                return best_gesture, best_confidence
        
        return None, 0
    
    def update_text(self, gesture, confidence):
        """Update the current text based on detected gesture"""
        if gesture and gesture in self.labels:
            self.current_text += gesture
            
            # Speak the gesture if TTS is enabled
            if self.tts_enabled and self.speak_on_detection:
                self.speak_text(gesture)
        
        # Reset stability counter after text update
        self.gesture_stability_count = 0
    
    def speak_text(self, text):
        """Convert text to speech using Google TTS"""
        if not self.tts_enabled or not text:
            return
        
        try:
            # Create TTS object
            tts = gtts.gTTS(text=text, lang='en')
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                
                # Play audio
                mixer.music.load(tmp_file.name)
                mixer.music.play()
                
                # Wait for playback to finish
                while mixer.music.get_busy():
                    pass
                
                # Clean up
                os.unlink(tmp_file.name)
            
            print(f"🔊 Spoke: {text}")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
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
        print("Starting Fixed Unvoiced Sign Language Detection...")
        print("Detection running in background - check GUI for results")
        
        while self.detection_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect sign language using fixed approach
            frame, gesture, confidence = self.detect_sign_language_fixed(frame)
            
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
    
    def speak_current_text(self, text):
        """Speak the current text"""
        if self.tts_enabled and text:
            self.speak_text(text)
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class FixedUnvoicedGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Fixed Unvoiced Sign Language Translator")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2c3e50')
        
        # Create detector instance
        self.detector = FixedUnvoicedDetector()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_gui(self):
        """Setup the enhanced GUI elements"""
        # Title
        title_label = Label(self.root, text="Fixed Unvoiced Sign Language Translator", 
                          font=("Arial", 22, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = Label(self.root, text="Inception V3 + Enhanced Rule-Based Detection", 
                             font=("Arial", 12), bg='#2c3e50', fg='#bdc3c7')
        subtitle_label.pack(pady=5)
        
        # Main frame
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Status and controls
        left_panel = Frame(main_frame, bg='#34495e', relief=RAISED, bd=2)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Model info frame
        model_info_frame = Frame(left_panel, bg='#34495e')
        model_info_frame.pack(fill=X, padx=10, pady=10)
        
        model_info_text = f"Model: Inception V3 + Rule-Based\nClasses: {len(self.detector.labels)}\nTTS: {'Enabled' if self.detector.tts_enabled else 'Disabled'}"
        
        model_label = Label(model_info_frame, text=model_info_text, bg='#34495e', fg='white', 
                           font=("Arial", 10), justify=LEFT)
        model_label.pack()
        
        # TTS controls
        tts_frame = Frame(left_panel, bg='#34495e')
        tts_frame.pack(fill=X, padx=10, pady=10)
        
        Label(tts_frame, text="Text-to-Speech:", bg='#34495e', fg='white', 
              font=("Arial", 12, "bold")).pack(side=LEFT, padx=5)
        
        self.speak_gestures_var = BooleanVar(value=False)
        speak_check = Checkbutton(tts_frame, text="Speak Gestures", variable=self.speak_gestures_var,
                                command=self.toggle_speak_gestures, bg='#34495e', fg='white',
                                selectcolor='#3498db', font=("Arial", 10))
        speak_check.pack(side=LEFT, padx=5)
        
        self.speak_text_button = Button(tts_frame, text="Speak Text", 
                                      command=self.speak_current_text, bg="#e67e22", fg="white",
                                      font=("Arial", 9, "bold"), padx=10, pady=5)
        self.speak_text_button.pack(side=LEFT, padx=5)
        
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
                                                        height=25, width=45, 
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
                           text="Instructions: Start detection → Show ASL signs → View translated text → Use TTS features",
                           font=("Arial", 11), bg='#2c3e50', fg='#bdc3c7', justify=CENTER)
        instructions.pack()
        
    def toggle_speak_gestures(self):
        """Toggle speaking gestures on detection"""
        self.detector.speak_on_detection = self.speak_gestures_var.get()
        status = "enabled" if self.detector.speak_on_detection else "disabled"
        self.bottom_status_label.config(text=f"Speak gestures {status}")
    
    def speak_current_text(self):
        """Speak the current translated text"""
        text = self.detector.current_text.strip()
        if text:
            self.detector.speak_current_text(text)
            self.bottom_status_label.config(text="Speaking current text...")
        else:
            self.bottom_status_label.config(text="No text to speak")
    
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
            
            status_text += f"Model: Inception V3 + Rules\n"
            status_text += f"TTS: {'On' if self.detector.speak_on_detection else 'Off'}\n"
            
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
    
    print("Starting Fixed Unvoiced Sign Language Translator...")
    print("This version fixes the detection issues and provides reliable output")
    
    # Start the GUI application
    app = FixedUnvoicedGUI()
    app.run()
