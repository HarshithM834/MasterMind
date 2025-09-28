import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext, messagebox
import os
import gtts
from pygame import mixer
import tempfile

class SimpleWorkingDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize text-to-speech
        self.tts_enabled = True
        try:
            mixer.init()
            print("✅ Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio system not available: {e}")
            self.tts_enabled = False
        
        # ASL alphabet
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        # Gesture recognition parameters
        self.gesture_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.6
        self.stability_frames = 5
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
        
    def classify_simple_gesture(self, landmarks):
        """Simple but effective gesture classification"""
        if landmarks is None:
            return None
        
        # Get finger tip and joint positions
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        mcp = [2, 5, 9, 13, 17]   # MCP joints
        pip = [3, 6, 10, 14, 18]  # PIP joints
        
        # Check finger extension status
        fingers_extended = []
        for i in range(5):
            if i == 0:  # Thumb (different logic)
                if landmarks[tips[i]].x > landmarks[mcp[i]].x:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
            else:  # Other fingers
                if landmarks[tips[i]].y < landmarks[mcp[i]].y:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
        
        extended_count = sum(fingers_extended)
        
        # Simple but effective gesture recognition
        if extended_count == 0:
            return 'A'  # Fist
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'B'  # Index finger
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'C'  # Index and middle
        elif extended_count == 3 and fingers_extended[1] == 1 and fingers_extended[2] == 1 and fingers_extended[3] == 1:
            return 'D'  # Three fingers
        elif extended_count == 4 and fingers_extended[0] == 0:
            return 'E'  # All fingers except thumb
        elif extended_count == 5:
            return 'F'  # All fingers
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'G'  # Thumb and index
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'I'  # Pinky
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[4] == 1:
            return 'Y'  # Thumb and pinky
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            # Check if fingers are close together (H) or apart (U/V)
            tip_distance = abs(landmarks[8].x - landmarks[12].x)
            if tip_distance < 0.05:
                return 'H'  # Close together
            else:
                return 'U'  # Apart
        else:
            return None
    
    def detect_sign_language(self, frame):
        """Main detection function"""
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
                
                # Classify gesture
                gesture = self.classify_simple_gesture(hand_landmarks.landmark)
                if gesture:
                    detected_gesture = gesture
                    confidence = 0.8  # High confidence for simple detection
                    self.gesture_buffer.append((gesture, confidence))
        
        return frame, detected_gesture, confidence
    
    def process_gesture_buffer(self):
        """Process gesture buffer with stability checking"""
        if len(self.gesture_buffer) < self.stability_frames:
            return None, 0
        
        # Get most recent gestures
        recent_gestures = list(self.gesture_buffer)[-self.stability_frames:]
        
        # Count gesture occurrences
        gesture_counts = {}
        for gesture, confidence in recent_gestures:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        if not gesture_counts:
            return None, 0
        
        # Find most common gesture
        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        confidence = gesture_counts[most_common_gesture] / len(recent_gestures)
        
        # Check stability
        if confidence >= self.confidence_threshold:
            if most_common_gesture == self.last_gesture:
                self.gesture_stability_count += 1
            else:
                self.gesture_stability_count = 1
                self.last_gesture = most_common_gesture
            
            # Require gesture to be stable for a few frames
            if self.gesture_stability_count >= 3:
                return most_common_gesture, confidence
        
        return None, 0
    
    def update_text(self, gesture, confidence):
        """Update the current text based on detected gesture"""
        if gesture and gesture in self.labels:
            self.current_text += gesture
            
            # Speak the gesture if TTS is enabled
            if self.tts_enabled and self.speak_on_detection:
                self.speak_text(gesture)
            
            print(f"✅ Detected: {gesture} (confidence: {confidence:.2f})")
        
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
        print("Starting Simple Sign Language Detection...")
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
    
    def speak_current_text(self, text):
        """Speak the current text"""
        if self.tts_enabled and text:
            self.speak_text(text)
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class SimpleWorkingGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Simple Working Sign Language Translator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Create detector instance
        self.detector = SimpleWorkingDetector()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_gui(self):
        """Setup the GUI elements"""
        # Title
        title_label = Label(self.root, text="Simple Working Sign Language Translator", 
                          font=("Arial", 22, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = Label(self.root, text="MediaPipe + Simple Rule-Based Detection", 
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
        
        model_info_text = f"Model: MediaPipe + Rule-Based\nClasses: {len(self.detector.labels)}\nTTS: {'Enabled' if self.detector.tts_enabled else 'Disabled'}"
        
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
            
            status_text += f"Model: MediaPipe + Rules\n"
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
    
    print("Starting Simple Working Sign Language Translator...")
    print("This version uses MediaPipe with simple rule-based detection for reliable output")
    
    # Start the GUI application
    app = SimpleWorkingGUI()
    app.run()
