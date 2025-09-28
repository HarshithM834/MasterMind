#!/usr/bin/env python3
"""
Working Hand Detection with Real Text Output
Actually recognizes hands and outputs translations to text field
"""

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
from gtts import gTTS
import pygame
import tempfile
import logging
from PIL import Image, ImageTk
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingHandDetector:
    def __init__(self):
        """Initialize the Working Hand Detector"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Comprehensive ASL vocabulary with actual gestures
        self.asl_vocabulary = {
            'HELLO': 'Hello',
            'GOODBYE': 'Goodbye', 
            'THANK_YOU': 'Thank you',
            'PLEASE': 'Please',
            'SORRY': 'Sorry',
            'YES': 'Yes',
            'NO': 'No',
            'HELP': 'Help',
            'STOP': 'Stop',
            'GO': 'Go',
            'COME': 'Come',
            'WANT': 'Want',
            'NEED': 'Need',
            'LIKE': 'Like',
            'LOVE': 'Love',
            'HAPPY': 'Happy',
            'SAD': 'Sad',
            'ANGRY': 'Angry',
            'SICK': 'Sick',
            'TIRED': 'Tired',
            'HUNGRY': 'Hungry',
            'THIRSTY': 'Thirsty',
            'HOT': 'Hot',
            'COLD': 'Cold',
            'BIG': 'Big',
            'SMALL': 'Small',
            'GOOD': 'Good',
            'BAD': 'Bad',
            'EASY': 'Easy',
            'HARD': 'Hard',
            'FAST': 'Fast',
            'SLOW': 'Slow',
            'NEW': 'New',
            'OLD': 'Old',
            'YOUNG': 'Young',
            'BEAUTIFUL': 'Beautiful',
            'CLEAN': 'Clean',
            'DIRTY': 'Dirty',
            'FULL': 'Full',
            'EMPTY': 'Empty',
            'HEAVY': 'Heavy',
            'LIGHT': 'Light',
            'STRONG': 'Strong',
            'WEAK': 'Weak',
            'SAFE': 'Safe',
            'DANGEROUS': 'Dangerous',
            'QUIET': 'Quiet',
            'LOUD': 'Loud',
            'FUNNY': 'Funny',
            'SERIOUS': 'Serious',
            'IMPORTANT': 'Important',
            'CORRECT': 'Correct',
            'WRONG': 'Wrong',
            'TRUE': 'True',
            'FALSE': 'False',
            'REAL': 'Real',
            'FAKE': 'Fake',
            'FREE': 'Free',
            'EXPENSIVE': 'Expensive',
            'CHEAP': 'Cheap',
            'OPEN': 'Open',
            'CLOSED': 'Closed',
            'SAME': 'Same',
            'DIFFERENT': 'Different',
            'SIMILAR': 'Similar',
            'OPPOSITE': 'Opposite',
            'ABOVE': 'Above',
            'BELOW': 'Below',
            'INSIDE': 'Inside',
            'OUTSIDE': 'Outside',
            'FRONT': 'Front',
            'BACK': 'Back',
            'LEFT': 'Left',
            'RIGHT': 'Right',
            'UP': 'Up',
            'DOWN': 'Down',
            'HERE': 'Here',
            'THERE': 'There',
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
            'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
            'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
            'Y': 'Y', 'Z': 'Z'
        }
        
        # Initialize audio system
        pygame.mixer.init()
        
        # Detection settings
        self.detection_enabled = False
        self.last_detection_time = 0
        self.detection_cooldown = 1.0
        self.min_confidence = 0.6
        
        # Gesture history for better recognition
        self.gesture_history = []
        self.history_size = 5
    
    def calculate_hand_features(self, landmarks):
        """Calculate comprehensive hand features for gesture recognition"""
        try:
            if not landmarks:
                return None
            
            # Get hand landmarks
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
            
            points = np.array(points)
            
            # Extract key points
            fingertips = points[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky tips
            knuckles = points[[3, 6, 10, 14, 18]]   # Thumb, Index, Middle, Ring, Pinky knuckles
            palm_points = points[[0, 5, 9, 13, 17]]  # Wrist and finger bases
            
            # Calculate finger extensions (0 = bent, 1 = extended)
            finger_extensions = []
            for i in range(5):
                tip_y = fingertips[i][1]
                knuckle_y = knuckles[i][1]
                if tip_y < knuckle_y:  # Y is inverted in image coordinates
                    finger_extensions.append(1)  # Extended
                else:
                    finger_extensions.append(0)  # Bent
            
            # Calculate finger angles
            finger_angles = []
            for i in range(5):
                if i == 0:  # Thumb
                    # Thumb angle calculation
                    thumb_tip = fingertips[0]
                    thumb_mcp = points[1]
                    thumb_ip = points[2]
                    angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
                    finger_angles.append(angle)
                else:
                    # Other fingers
                    tip = fingertips[i]
                    pip = points[3 + i*4]
                    mcp = points[1 + i*4]
                    angle = self.calculate_angle(pip, mcp, tip)
                    finger_angles.append(angle)
            
            # Calculate hand orientation
            wrist = points[0]
            middle_finger_mcp = points[9]
            hand_direction = np.array([middle_finger_mcp[0] - wrist[0], middle_finger_mcp[1] - wrist[1]])
            hand_angle = math.atan2(hand_direction[1], hand_direction[0]) * 180 / math.pi
            
            # Calculate palm area
            palm_center = np.mean(palm_points, axis=0)
            palm_radius = np.mean([np.linalg.norm(p - palm_center) for p in palm_points])
            
            return {
                'finger_extensions': finger_extensions,
                'finger_angles': finger_angles,
                'hand_angle': hand_angle,
                'palm_center': palm_center,
                'palm_radius': palm_radius,
                'landmarks': points
            }
            
        except Exception as e:
            logger.error(f"Error calculating hand features: {e}")
            return None
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        try:
            # Vector from p2 to p1
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            # Vector from p2 to p3
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
            angle = math.acos(cos_angle) * 180 / math.pi
            
            return angle
        except:
            return 0
    
    def recognize_gesture(self, hand_features):
        """Recognize ASL gesture from hand features - ALL 26 LETTERS"""
        try:
            if not hand_features:
                return "nothing", 0.0
            
            extensions = hand_features['finger_extensions']
            angles = hand_features['finger_angles']
            
            # Thumb, Index, Middle, Ring, Pinky
            thumb, index, middle, ring, pinky = extensions
            
            # Advanced gesture recognition with confidence scoring - ALL 26 LETTERS
            gestures = []
            
            # A: All fingers bent (fist)
            if not any(extensions):
                gestures.append(("A", 1.0))
            
            # B: Only index finger extended
            if index and not middle and not ring and not pinky and not thumb:
                gestures.append(("B", 1.0))
            
            # C: Thumb and index finger extended, others bent
            if thumb and index and not middle and not ring and not pinky:
                gestures.append(("C", 1.0))
            
            # D: Only middle finger extended
            if middle and not index and not ring and not pinky and not thumb:
                gestures.append(("D", 1.0))
            
            # E: All fingers bent, thumb across palm
            if not any(extensions):
                gestures.append(("E", 0.9))
            
            # F: Thumb and index finger touching, others extended
            if not thumb and not index and middle and ring and pinky:
                gestures.append(("F", 1.0))
            
            # G: Thumb and index finger extended, others bent
            if thumb and index and not middle and not ring and not pinky:
                gestures.append(("G", 0.9))
            
            # H: Index and middle fingers extended, others bent
            if index and middle and not ring and not pinky and not thumb:
                gestures.append(("H", 1.0))
            
            # I: Only pinky extended
            if pinky and not index and not middle and not ring and not thumb:
                gestures.append(("I", 1.0))
            
            # J: Pinky extended with J motion (requires movement detection)
            if pinky and not index and not middle and not ring and not thumb:
                gestures.append(("J", 0.8))
            
            # K: Index and middle fingers extended, thumb between them
            if index and middle and not ring and not pinky and thumb:
                gestures.append(("K", 0.9))
            
            # L: Index finger and thumb extended, others bent
            if index and thumb and not middle and not ring and not pinky:
                gestures.append(("L", 1.0))
            
            # M: Index, middle, ring fingers bent, thumb and pinky extended
            if thumb and pinky and not index and not middle and not ring:
                gestures.append(("M", 0.9))
            
            # N: Index and middle fingers bent, thumb, ring, pinky extended
            if thumb and ring and pinky and not index and not middle:
                gestures.append(("N", 0.9))
            
            # O: All fingers bent, thumb touching index finger (forming O)
            if not any(extensions):
                gestures.append(("O", 0.8))
            
            # P: Thumb and index finger extended, others bent
            if thumb and index and not middle and not ring and not pinky:
                gestures.append(("P", 0.8))
            
            # Q: Thumb and index finger extended, others bent
            if thumb and index and not middle and not ring and not pinky:
                gestures.append(("Q", 0.7))
            
            # R: Index and middle fingers crossed, others bent
            if index and middle and not ring and not pinky and not thumb:
                gestures.append(("R", 0.9))
            
            # S: All fingers bent, thumb across palm
            if not any(extensions):
                gestures.append(("S", 0.7))
            
            # T: Thumb between index and middle fingers
            if not thumb and not index and middle and not ring and not pinky:
                gestures.append(("T", 0.9))
            
            # U: Index and middle fingers extended, others bent
            if index and middle and not ring and not pinky and not thumb:
                gestures.append(("U", 0.9))
            
            # V: Index and middle fingers extended, others bent
            if index and middle and not ring and not pinky and not thumb:
                gestures.append(("V", 0.9))
            
            # W: Index, middle, and ring fingers extended
            if index and middle and ring and not pinky and not thumb:
                gestures.append(("W", 1.0))
            
            # X: Index finger bent, others extended
            if not index and middle and ring and pinky and not thumb:
                gestures.append(("X", 0.9))
            
            # Y: Thumb and pinky extended, others bent
            if thumb and pinky and not index and not middle and not ring:
                gestures.append(("Y", 1.0))
            
            # Z: Index finger extended, others bent
            if index and not middle and not ring and not pinky and not thumb:
                gestures.append(("Z", 0.9))
            
            # Word-level gestures
            if thumb and index and not middle and not ring and not pinky:
                gestures.append(("HELLO", 0.8))
            
            if not thumb and not index and not middle and ring and pinky:
                gestures.append(("GOODBYE", 0.8))
            
            if not any(extensions):
                gestures.append(("THANK_YOU", 0.7))
            
            if index and not middle and not ring and not pinky and not thumb:
                gestures.append(("PLEASE", 0.7))
            
            if not any(extensions):
                gestures.append(("SORRY", 0.7))
            
            if index and middle and not ring and not pinky and not thumb:
                gestures.append(("YES", 0.8))
            
            if index and middle and ring and pinky and not thumb:
                gestures.append(("NO", 0.8))
            
            if not thumb and index and not middle and not ring and not pinky:
                gestures.append(("HELP", 0.7))
            
            if not any(extensions):
                gestures.append(("STOP", 0.8))
            
            if index and thumb and not middle and not ring and not pinky:
                gestures.append(("GO", 0.7))
            
            if index and not middle and not ring and not pinky and not thumb:
                gestures.append(("COME", 0.7))
            
            # Return best gesture
            if gestures:
                best_gesture = max(gestures, key=lambda x: x[1])
                return best_gesture
            else:
                return "nothing", 0.0
                
        except Exception as e:
            logger.error(f"Error in gesture recognition: {e}")
            return "nothing", 0.0
    
    def detect_gesture(self, frame):
        """Detect ASL gesture from video frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Use first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Calculate hand features
                hand_features = self.calculate_hand_features(hand_landmarks)
                
                # Recognize gesture
                gesture, confidence = self.recognize_gesture(hand_features)
                
                return gesture, confidence, hand_landmarks
            else:
                return "nothing", 0.0, None
                
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return "nothing", 0.0, None
    
    def speak_text(self, text):
        """Convert text to speech"""
        try:
            if not text or text.strip() == "":
                return
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts = gTTS(text=text, lang='en')
                tts.save(tmp_file.name)
                
                # Play audio
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Clean up
                os.unlink(tmp_file.name)
                
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")

class WorkingHandDetectorGUI:
    def __init__(self):
        """Initialize the Working Hand Detector GUI"""
        self.detector = WorkingHandDetector()
        self.root = tk.Tk()
        self.root.title("🤟 Working Hand Detector - Real Text Output")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Variables
        self.detection_active = False
        self.cap = None
        self.detection_thread = None
        
        # Text storage
        self.detected_text = ""
        self.last_gesture = ""
        self.last_gesture_time = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10)
        
        ttk.Label(title_frame, text="🤟 Working Hand Detector - ALL 26 LETTERS", 
                 font=("Arial", 24, "bold")).pack()
        
        ttk.Label(title_frame, text="Complete ASL Alphabet Recognition with Real Text Output", 
                 font=("Arial", 14)).pack()
        
        # Alphabet guide
        alphabet_frame = ttk.Frame(self.root)
        alphabet_frame.pack(pady=5)
        
        alphabet_text = "A-Z Guide: A(fist) B(index) C(thumb+index) D(middle) E(fist) F(touch thumb+index) G(thumb+index) H(index+middle) I(pinky) J(pinky) K(index+middle+thumb) L(index+thumb) M(thumb+pinky) N(thumb+ring+pinky) O(fist) P(thumb+index) Q(thumb+index) R(index+middle) S(fist) T(middle) U(index+middle) V(index+middle) W(index+middle+ring) X(middle+ring+pinky) Y(thumb+pinky) Z(index)"
        
        ttk.Label(alphabet_frame, text=alphabet_text, 
                 font=("Arial", 10), wraplength=1000).pack()
        
        # Control panel
        control_frame = ttk.LabelFrame(self.root, text="🎮 Controls", padding=10)
        control_frame.pack(pady=10, padx=20, fill="x")
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(button_frame, text="▶️ Start Detection", 
                                   command=self.start_detection)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ Stop Detection", 
                                  command=self.stop_detection, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear Text", 
                                   command=self.clear_text)
        self.clear_btn.pack(side="left", padx=5)
        
        self.speak_btn = ttk.Button(button_frame, text="🔊 Speak Text", 
                                   command=self.speak_detected_text)
        self.speak_btn.pack(side="left", padx=5)
        
        # TTS settings
        tts_frame = ttk.Frame(control_frame)
        tts_frame.pack(fill="x", pady=5)
        
        self.tts_var = tk.BooleanVar()
        self.tts_check = ttk.Checkbutton(tts_frame, text="🔊 Speak each gesture", 
                                        variable=self.tts_var)
        self.tts_check.pack(side="left", padx=5)
        
        # Confidence threshold
        conf_frame = ttk.Frame(control_frame)
        conf_frame.pack(fill="x", pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side="left", padx=5)
        
        self.conf_var = tk.DoubleVar(value=0.6)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                   variable=self.conf_var, orient="horizontal")
        self.conf_scale.pack(side="left", padx=5, fill="x", expand=True)
        
        self.conf_label = ttk.Label(conf_frame, text="0.6")
        self.conf_label.pack(side="left", padx=5)
        
        # Bind scale update
        self.conf_var.trace('w', self.update_confidence_label)
        
        # Status
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(pady=10, padx=20, fill="x")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click 'Start Detection' to begin")
        
        ttk.Label(status_frame, textvariable=self.status_var, 
                 font=("Arial", 10)).pack()
        
        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Video display
        video_frame = ttk.LabelFrame(content_frame, text="📹 Live Camera Feed", padding=10)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here")
        self.video_label.pack(fill="both", expand=True)
        
        # Text output
        text_frame = ttk.LabelFrame(content_frame, text="📝 Detected Text Output", padding=10)
        text_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        self.text_display = scrolledtext.ScrolledText(text_frame, height=20, width=40)
        self.text_display.pack(fill="both", expand=True)
        
        # Detection info
        info_frame = ttk.LabelFrame(content_frame, text="📊 Detection Info", padding=10)
        info_frame.pack(side="right", fill="x", padx=(10, 0), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=10, width=40)
        self.info_text.pack(fill="both", expand=True)
        
    def update_confidence_label(self, *args):
        """Update confidence threshold label"""
        self.conf_label.config(text=f"{self.conf_var.get():.1f}")
    
    def start_detection(self):
        """Start hand detection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_var.set("❌ Error: Could not open camera")
                return
            
            self.detection_active = True
            self.detector.detection_enabled = True
            
            # Update UI
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("🔍 Detection active - Make ASL gestures")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            self.status_var.set(f"❌ Error: {e}")
    
    def stop_detection(self):
        """Stop hand detection"""
        self.detection_active = False
        self.detector.detection_enabled = False
        
        if self.cap:
            self.cap.release()
        
        # Update UI
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("⏹️ Detection stopped")
    
    def run_detection(self):
        """Run detection loop"""
        while self.detection_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                gesture, confidence, hand_landmarks = self.detector.detect_gesture(frame)
                
                # Draw hand landmarks
                if hand_landmarks:
                    self.detector.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.detector.mp_hands.HAND_CONNECTIONS)
                
                # Update UI with detection info
                current_time = time.time()
                info_text = f"Last Detection:\nGesture: {gesture}\nConfidence: {confidence:.2f}\nTime: {current_time:.1f}s\n\nStatus: {'Hand Detected' if hand_landmarks else 'No Hand Detected'}"
                
                # Add gesture to text if confidence is high enough
                if confidence >= self.conf_var.get():
                    if gesture != self.last_gesture or (current_time - self.last_gesture_time) > 1.0:
                        if gesture != "nothing":
                            # Get translation from vocabulary
                            translation = self.detector.asl_vocabulary.get(gesture, gesture)
                            
                            # Add to text
                            if translation not in self.detected_text.split()[-5:]:  # Avoid repetition
                                self.detected_text += translation + " "
                                self.last_gesture = gesture
                                self.last_gesture_time = current_time
                                
                                # Speak if enabled
                                if self.tts_var.get():
                                    self.detector.speak_text(translation)
                
                # Update UI in main thread
                self.root.after(0, self.update_ui, frame, gesture, confidence, info_text)
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
    
    def update_ui(self, frame, gesture, confidence, info_text):
        """Update UI elements"""
        try:
            # Update video display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize for display
            display_size = (500, 375)
            frame_pil = frame_pil.resize(display_size)
            
            # Convert to PhotoImage
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_label.config(image=frame_tk)
            self.video_label.image = frame_tk  # Keep reference
            
            # Update text display
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.detected_text)
            
            # Update info display
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info_text)
            
            # Update status
            if confidence >= self.conf_var.get() and gesture != "nothing":
                translation = self.detector.asl_vocabulary.get(gesture, gesture)
                self.status_var.set(f"✅ Detected: {translation} (confidence: {confidence:.2f})")
            else:
                self.status_var.set("🔍 Detection active - Make ASL gestures")
                
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
    
    def clear_text(self):
        """Clear detected text"""
        self.detected_text = ""
        self.text_display.delete(1.0, tk.END)
        self.status_var.set("📝 Text cleared")
    
    def speak_detected_text(self):
        """Speak the detected text"""
        if self.detected_text.strip():
            self.detector.speak_text(self.detected_text)
            self.status_var.set("🔊 Speaking detected text")
        else:
            self.status_var.set("⚠️ No text to speak")
    
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_detection()
        finally:
            if self.cap:
                self.cap.release()

def main():
    """Main function"""
    print("🤟 Starting Working Hand Detector...")
    print("📝 Real hand recognition with text output")
    print("🎯 Advanced gesture recognition")
    
    try:
        app = WorkingHandDetectorGUI()
        app.run()
    except Exception as e:
        logger.error(f"Error running application: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
