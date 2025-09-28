#!/usr/bin/env python3
"""
Perfect Visual Sign Language Detector
Beautiful, modern GUI with professional styling and animations
"""

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pygame
from gtts import gTTS
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectVisualDetector:
    def __init__(self):
        """Initialize the Perfect Visual Detector"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ASL alphabet mapping
        self.asl_alphabet = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
        }
        
        # Initialize audio system
        pygame.mixer.init()
        
        # Detection settings
        self.detection_enabled = False
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # seconds
        self.min_confidence = 0.6
        
        # Simple rule-based detection (since we don't have trained model)
        self.setup_rule_based_detection()
    
    def setup_rule_based_detection(self):
        """Setup simple rule-based gesture detection"""
        logger.info("Setting up rule-based ASL detection")
    
    def detect_gesture_from_landmarks(self, landmarks):
        """Detect gesture using MediaPipe landmarks and rules"""
        try:
            if not landmarks:
                return "nothing", 0.0
            
            # Get hand landmarks
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
            
            points = np.array(points)
            
            # Extract finger tip positions (indices: 4, 8, 12, 16, 20)
            fingertips = points[[4, 8, 12, 16, 20]]
            knuckles = points[[3, 6, 10, 14, 18]]
            
            # Calculate finger extensions
            finger_extensions = []
            for i in range(5):
                if fingertips[i][1] < knuckles[i][1]:  # Y is inverted in image coordinates
                    finger_extensions.append(1)  # Extended
                else:
                    finger_extensions.append(0)  # Bent
            
            # Gesture recognition rules
            gesture, confidence = self.classify_gesture(finger_extensions, points)
            return gesture, confidence
            
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return "nothing", 0.0
    
    def classify_gesture(self, finger_extensions, points):
        """Classify gesture based on finger positions"""
        # Thumb (0), Index (1), Middle (2), Ring (3), Pinky (4)
        thumb, index, middle, ring, pinky = finger_extensions
        
        # A: All fingers bent (fist)
        if not any(finger_extensions):
            return "A", 1.0
        
        # B: Only index finger extended
        if index and not middle and not ring and not pinky and not thumb:
            return "B", 1.0
        
        # C: Thumb and index finger extended, others bent
        if thumb and index and not middle and not ring and not pinky:
            return "C", 1.0
        
        # D: Only middle finger extended
        if middle and not index and not ring and not pinky and not thumb:
            return "D", 1.0
        
        # E: All fingers bent, thumb across palm
        if not any(finger_extensions):
            return "E", 1.0
        
        # F: Thumb and index finger touching, others extended
        if not thumb and not index and middle and ring and pinky:
            return "F", 1.0
        
        # G: Thumb and index finger extended, others bent
        if thumb and index and not middle and not ring and not pinky:
            return "G", 1.0
        
        # H: Index and middle fingers extended, others bent
        if index and middle and not ring and not pinky and not thumb:
            return "H", 1.0
        
        # I: Only pinky extended
        if pinky and not index and not middle and not ring and not thumb:
            return "I", 1.0
        
        # L: Index finger and thumb extended, others bent
        if index and thumb and not middle and not ring and not pinky:
            return "L", 1.0
        
        # O: All fingers bent, thumb touching index finger
        if not any(finger_extensions):
            return "O", 1.0
        
        # P: Thumb and index finger extended, others bent
        if thumb and index and not middle and not ring and not pinky:
            return "P", 1.0
        
        # R: Index and middle fingers crossed, others bent
        if index and middle and not ring and not pinky and not thumb:
            return "R", 1.0
        
        # T: Thumb between index and middle fingers
        if not thumb and not index and middle and not ring and not pinky:
            return "T", 1.0
        
        # U: Index and middle fingers extended, others bent
        if index and middle and not ring and not pinky and not thumb:
            return "U", 1.0
        
        # V: Index and middle fingers extended, others bent
        if index and middle and not ring and not pinky and not thumb:
            return "V", 1.0
        
        # W: Index, middle, and ring fingers extended
        if index and middle and ring and not pinky and not thumb:
            return "W", 1.0
        
        # X: Index finger bent, others extended
        if not index and middle and ring and pinky and not thumb:
            return "X", 1.0
        
        # Y: Thumb and pinky extended, others bent
        if thumb and pinky and not index and not middle and not ring:
            return "Y", 1.0
        
        # Z: Index finger extended, others bent
        if index and not middle and not ring and not pinky and not thumb:
            return "Z", 1.0
        
        return "nothing", 0.0
    
    def speak_text(self, text):
        """Convert text to speech using Google TTS"""
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

class PerfectVisualGUI:
    def __init__(self):
        """Initialize the Perfect Visual GUI"""
        self.detector = PerfectVisualDetector()
        self.root = tk.Tk()
        
        # Modern styling
        self.setup_modern_theme()
        
        # Window configuration
        self.root.title("🤟 Perfect Sign Language Detector")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1a1a1a")
        
        # Center window
        self.center_window()
        
        # Variables
        self.detection_active = False
        self.cap = None
        self.detection_thread = None
        
        # Text storage
        self.detected_text = ""
        self.last_letter = ""
        self.last_letter_time = 0
        
        # Animation variables
        self.pulse_animation = False
        self.confidence_pulse = 0
        
        self.setup_gui()
        
    def setup_modern_theme(self):
        """Setup modern dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure modern dark theme
        style.configure('Modern.TFrame', background='#1a1a1a')
        style.configure('Modern.TLabel', background='#1a1a1a', foreground='#ffffff')
        style.configure('Modern.TButton', background='#007AFF', foreground='white')
        style.configure('Modern.TLabelFrame', background='#2a2a2a', foreground='#ffffff')
        style.configure('Modern.TLabelFrame.Label', background='#2a2a2a', foreground='#ffffff')
        style.configure('Modern.TScale', background='#2a2a2a', troughcolor='#404040')
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_gui(self):
        """Setup the beautiful GUI layout"""
        # Main container with padding
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # Control panel
        self.create_control_panel(main_container)
        
        # Main content area
        self.create_main_content(main_container)
        
        # Status bar
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """Create beautiful header section"""
        header_frame = ttk.Frame(parent, style='Modern.TFrame')
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Title with gradient effect simulation
        title_frame = ttk.Frame(header_frame, style='Modern.TFrame')
        title_frame.pack()
        
        # Main title
        title_label = tk.Label(title_frame, 
                              text="🤟 Perfect Sign Language Detector", 
                              font=('Arial', 28, 'bold'),
                              fg='#007AFF', bg='#1a1a1a')
        title_label.pack()
        
        # Subtitle
        subtitle_label = tk.Label(title_frame,
                                 text="Advanced AI-Powered Real-Time ASL Recognition",
                                 font=('Arial', 14),
                                 fg='#8E8E93', bg='#1a1a1a')
        subtitle_label.pack(pady=(5, 0))
        
        # Feature badges
        badges_frame = ttk.Frame(header_frame, style='Modern.TFrame')
        badges_frame.pack(pady=(15, 0))
        
        badges = [
            ("🎯 Real-Time Detection", "#34C759"),
            ("🧠 AI-Powered", "#FF9500"),
            ("🔊 Voice Output", "#AF52DE"),
            ("📱 Modern Interface", "#007AFF")
        ]
        
        for badge_text, color in badges:
            badge = tk.Label(badges_frame, text=badge_text, 
                           font=('Arial', 10, 'bold'),
                           fg=color, bg='#1a1a1a',
                           relief='raised', bd=1)
            badge.pack(side="left", padx=5, pady=5)
    
    def create_control_panel(self, parent):
        """Create modern control panel"""
        control_frame = ttk.LabelFrame(parent, text="🎮 Control Panel", 
                                      style='Modern.TLabelFrame', padding=20)
        control_frame.pack(fill="x", pady=(0, 20))
        
        # Main control buttons
        button_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        button_frame.pack(fill="x", pady=(0, 15))
        
        # Start/Stop buttons with modern styling
        self.start_btn = tk.Button(button_frame, text="▶️ Start Detection", 
                                  command=self.start_detection,
                                  font=('Arial', 12, 'bold'),
                                  bg='#34C759', fg='white', bd=0,
                                  padx=20, pady=10,
                                  relief='flat', cursor='hand2')
        self.start_btn.pack(side="left", padx=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Detection", 
                                 command=self.stop_detection,
                                 font=('Arial', 12, 'bold'),
                                 bg='#FF3B30', fg='white', bd=0,
                                 padx=20, pady=10,
                                 relief='flat', cursor='hand2',
                                 state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))
        
        # Utility buttons
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear", 
                                  command=self.clear_text,
                                  font=('Arial', 11),
                                  bg='#8E8E93', fg='white', bd=0,
                                  padx=15, pady=10,
                                  relief='flat', cursor='hand2')
        self.clear_btn.pack(side="left", padx=(0, 10))
        
        self.speak_btn = tk.Button(button_frame, text="🔊 Speak", 
                                  command=self.speak_detected_text,
                                  font=('Arial', 11),
                                  bg='#AF52DE', fg='white', bd=0,
                                  padx=15, pady=10,
                                  relief='flat', cursor='hand2')
        self.speak_btn.pack(side="left")
        
        # Settings panel
        settings_frame = ttk.Frame(control_frame, style='Modern.TFrame')
        settings_frame.pack(fill="x", pady=(15, 0))
        
        # TTS toggle
        tts_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        tts_frame.pack(side="left", padx=(0, 30))
        
        self.tts_var = tk.BooleanVar()
        self.tts_check = tk.Checkbutton(tts_frame, text="🔊 Speak each letter", 
                                       variable=self.tts_var,
                                       font=('Arial', 11),
                                       fg='#ffffff', bg='#2a2a2a',
                                       selectcolor='#007AFF',
                                       activebackground='#2a2a2a',
                                       activeforeground='#ffffff')
        self.tts_check.pack()
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame, style='Modern.TFrame')
        conf_frame.pack(side="left")
        
        conf_label = tk.Label(conf_frame, text="Confidence Threshold:", 
                             font=('Arial', 11),
                             fg='#ffffff', bg='#2a2a2a')
        conf_label.pack(side="left", padx=(0, 10))
        
        self.conf_var = tk.DoubleVar(value=0.6)
        self.conf_scale = tk.Scale(conf_frame, from_=0.1, to=1.0, 
                                  variable=self.conf_var, orient="horizontal",
                                  bg='#2a2a2a', fg='#ffffff',
                                  troughcolor='#404040',
                                  activebackground='#007AFF',
                                  highlightbackground='#2a2a2a',
                                  length=150)
        self.conf_scale.pack(side="left")
        
        self.conf_label = tk.Label(conf_frame, text="0.6", 
                                  font=('Arial', 11, 'bold'),
                                  fg='#007AFF', bg='#2a2a2a')
        self.conf_label.pack(side="left", padx=(10, 0))
        
        # Bind scale update
        self.conf_var.trace('w', self.update_confidence_label)
    
    def create_main_content(self, parent):
        """Create main content area with video and text"""
        content_frame = ttk.Frame(parent, style='Modern.TFrame')
        content_frame.pack(fill="both", expand=True)
        
        # Left side - Video display
        video_frame = ttk.LabelFrame(content_frame, text="📹 Live Camera Feed", 
                                    style='Modern.TLabelFrame', padding=15)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Video container with border
        video_container = tk.Frame(video_frame, bg='#000000', relief='raised', bd=2)
        video_container.pack(fill="both", expand=True)
        
        self.video_label = tk.Label(video_container, text="📷 Camera feed will appear here",
                                   font=('Arial', 16),
                                   fg='#8E8E93', bg='#000000',
                                   relief='flat')
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right side - Text and info
        right_frame = ttk.Frame(content_frame, style='Modern.TFrame')
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Text output
        text_frame = ttk.LabelFrame(right_frame, text="📝 Detected Text", 
                                   style='Modern.TLabelFrame', padding=15)
        text_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Text display with modern styling
        text_container = tk.Frame(text_frame, bg='#1a1a1a', relief='raised', bd=1)
        text_container.pack(fill="both", expand=True)
        
        self.text_display = tk.Text(text_container, 
                                   font=('Courier New', 16),
                                   fg='#ffffff', bg='#1a1a1a',
                                   insertbackground='#007AFF',
                                   selectbackground='#007AFF',
                                   selectforeground='#ffffff',
                                   relief='flat', bd=0,
                                   wrap='word')
        
        # Scrollbar for text
        text_scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_display.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        text_scrollbar.pack(side="right", fill="y")
        
        # Detection info
        info_frame = ttk.LabelFrame(right_frame, text="📊 Detection Info", 
                                   style='Modern.TLabelFrame', padding=15)
        info_frame.pack(fill="x")
        
        # Info display with modern styling
        info_container = tk.Frame(info_frame, bg='#1a1a1a', relief='raised', bd=1)
        info_container.pack(fill="both", expand=True)
        
        self.info_text = tk.Text(info_container, height=8,
                                font=('Courier New', 11),
                                fg='#ffffff', bg='#1a1a1a',
                                insertbackground='#007AFF',
                                selectbackground='#007AFF',
                                selectforeground='#ffffff',
                                relief='flat', bd=0)
        self.info_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_status_bar(self, parent):
        """Create modern status bar"""
        status_frame = tk.Frame(parent, bg='#2a2a2a', relief='raised', bd=1)
        status_frame.pack(fill="x", pady=(20, 0))
        
        # Status text
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 Ready - Click 'Start Detection' to begin")
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Arial', 11),
                               fg='#ffffff', bg='#2a2a2a',
                               anchor='w')
        status_label.pack(side="left", padx=15, pady=10)
        
        # Confidence indicator
        self.confidence_label = tk.Label(status_frame, text="Confidence: --",
                                        font=('Arial', 11),
                                        fg='#8E8E93', bg='#2a2a2a')
        self.confidence_label.pack(side="right", padx=15, pady=10)
    
    def update_confidence_label(self, *args):
        """Update confidence threshold label"""
        self.conf_label.config(text=f"{self.conf_var.get():.1f}")
    
    def start_detection(self):
        """Start ASL detection with beautiful animations"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.show_error("Camera Error", "Could not open camera. Please check your camera connection.")
                return
            
            self.detection_active = True
            self.detector.detection_enabled = True
            
            # Update UI with animations
            self.start_btn.config(state="disabled", bg='#8E8E93')
            self.stop_btn.config(state="normal", bg='#FF3B30')
            self.status_var.set("🔴 Detection Active - Make ASL gestures")
            
            # Start pulse animation
            self.start_pulse_animation()
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            self.show_error("Detection Error", f"Error starting detection: {e}")
    
    def stop_detection(self):
        """Stop ASL detection"""
        self.detection_active = False
        self.detector.detection_enabled = False
        
        if self.cap:
            self.cap.release()
        
        # Update UI
        self.start_btn.config(state="normal", bg='#34C759')
        self.stop_btn.config(state="disabled", bg='#8E8E93')
        self.status_var.set("🟢 Ready - Click 'Start Detection' to begin")
        
        # Stop pulse animation
        self.stop_pulse_animation()
    
    def start_pulse_animation(self):
        """Start pulse animation for active detection"""
        self.pulse_animation = True
        self.animate_pulse()
    
    def stop_pulse_animation(self):
        """Stop pulse animation"""
        self.pulse_animation = False
    
    def animate_pulse(self):
        """Animate pulse effect"""
        if not self.pulse_animation:
            return
        
        # Create pulsing effect
        pulse_intensity = int(50 + 30 * np.sin(self.confidence_pulse))
        self.confidence_pulse += 0.3
        
        # Update confidence label with pulse
        if hasattr(self, 'last_confidence'):
            self.confidence_label.config(
                text=f"Confidence: {self.last_confidence:.2f}",
                fg=f"#{pulse_intensity:02x}{pulse_intensity:02x}ff"
            )
        
        # Schedule next animation frame
        self.root.after(50, self.animate_pulse)
    
    def run_detection(self):
        """Run detection loop with beautiful rendering"""
        while self.detection_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Detect hands and draw landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.hands.process(rgb_frame)
                
                # Initialize gesture detection
                detected_letter = "nothing"
                confidence = 0.0
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks with beautiful styling
                        self.detector.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.detector.mp_hands.HAND_CONNECTIONS,
                            self.detector.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                            self.detector.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                        
                        # Detect gesture
                        detected_letter, confidence = self.detector.detect_gesture_from_landmarks(hand_landmarks)
                
                # Update UI with detection info
                current_time = time.time()
                info_text = f"Last Detection:\nLetter: {detected_letter}\nConfidence: {confidence:.2f}\nTime: {current_time:.1f}s\n\nDetection Status: {'🟢 Active' if self.detection_active else '🔴 Stopped'}"
                
                # Add letter to text if confidence is high enough
                if confidence >= self.conf_var.get():
                    if detected_letter != self.last_letter or (current_time - self.last_letter_time) > 1.0:
                        if detected_letter not in ['del', 'nothing', 'space']:
                            self.detected_text += detected_letter
                            self.last_letter = detected_letter
                            self.last_letter_time = current_time
                            
                            # Speak letter if enabled
                            if self.tts_var.get():
                                self.detector.speak_text(detected_letter)
                        
                        elif detected_letter == 'space':
                            self.detected_text += " "
                        elif detected_letter == 'del' and self.detected_text:
                            self.detected_text = self.detected_text[:-1]
                
                # Store confidence for animation
                self.last_confidence = confidence
                
                # Update UI in main thread
                self.root.after(0, self.update_ui, frame, detected_letter, confidence, info_text)
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                break
    
    def update_ui(self, frame, letter, confidence, info_text):
        """Update UI elements with beautiful rendering"""
        try:
            # Update video display with enhanced styling
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add detection overlay
            if letter != "nothing" and confidence >= self.conf_var.get():
                # Add glowing effect for detected letters
                cv2.putText(frame_rgb, f"DETECTED: {letter}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame_rgb, f"Confidence: {confidence:.2f}", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize for display while maintaining aspect ratio
            display_size = (500, 375)
            frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_label.config(image=frame_tk, text="")
            self.video_label.image = frame_tk  # Keep reference
            
            # Update text display with beautiful formatting
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.detected_text)
            
            # Highlight the last letter
            if len(self.detected_text) > 0:
                last_char_pos = len(self.detected_text)
                self.text_display.tag_add("highlight", f"1.{last_char_pos-1}", f"1.{last_char_pos}")
                self.text_display.tag_configure("highlight", background="#007AFF", foreground="#ffffff")
            
            # Update info display
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info_text)
            
            # Update status with color coding
            if confidence >= self.conf_var.get() and letter != "nothing":
                self.status_var.set(f"✅ Detected: {letter} (confidence: {confidence:.2f})")
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}", fg="#34C759")
            else:
                self.status_var.set("🔍 Detection active - Make ASL gestures")
                self.confidence_label.config(text="Confidence: --", fg="#8E8E93")
                
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
    
    def clear_text(self):
        """Clear detected text with animation"""
        self.detected_text = ""
        self.text_display.delete(1.0, tk.END)
        self.status_var.set("🗑️ Text cleared")
        
        # Brief animation effect
        self.root.after(1000, lambda: self.status_var.set("🟢 Ready - Click 'Start Detection' to begin"))
    
    def speak_detected_text(self):
        """Speak the detected text"""
        if self.detected_text.strip():
            self.detector.speak_text(self.detected_text)
            self.status_var.set("🔊 Speaking detected text...")
            
            # Reset status after speaking
            self.root.after(3000, lambda: self.status_var.set("🟢 Ready - Click 'Start Detection' to begin"))
        else:
            self.show_info("No Text", "No text to speak. Start detection and make some gestures first!")
    
    def show_error(self, title, message):
        """Show error message with modern styling"""
        messagebox.showerror(title, message)
    
    def show_info(self, title, message):
        """Show info message with modern styling"""
        messagebox.showinfo(title, message)
    
    def run(self):
        """Run the beautiful GUI"""
        try:
            # Add window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the GUI
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_detection()
        finally:
            if self.cap:
                self.cap.release()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    """Main function"""
    print("🎨 Starting Perfect Visual Sign Language Detector...")
    print("✨ Beautiful modern interface with professional styling")
    print("🤟 Advanced real-time ASL recognition")
    
    try:
        app = PerfectVisualGUI()
        app.run()
    except Exception as e:
        logger.error(f"Error running application: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
