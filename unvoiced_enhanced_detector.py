import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext, messagebox
import os
from pathlib import Path
from unvoiced_integration import UnvoicedIntegration

class UnvoicedEnhancedDetector:
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
        
        # Initialize Unvoiced integration
        print("🤟 Initializing Unvoiced integration...")
        self.unvoiced = UnvoicedIntegration()
        
        # Gesture recognition parameters
        self.gesture_buffer = deque(maxlen=15)
        self.confidence_threshold = 0.85
        self.stability_frames = 10
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
        self.tts_enabled = True
        self.speak_on_detection = False
        
    def extract_hand_region_unvoiced(self, frame, landmarks):
        """Extract hand region optimized for Unvoiced model"""
        if landmarks is None:
            return None
        
        # Get bounding box of hand
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding for better hand capture
        padding = 0.15
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
        # This matches the approach used in Unvoiced
        size = max(hand_region.shape[0], hand_region.shape[1])
        square_region = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Center the hand in the square
        start_x = (size - hand_region.shape[1]) // 2
        start_y = (size - hand_region.shape[0]) // 2
        square_region[start_y:start_y + hand_region.shape[0], 
                     start_x:start_x + hand_region.shape[1]] = hand_region
        
        return square_region
    
    def detect_sign_language_unvoiced(self, frame):
        """Main detection function using Unvoiced model"""
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
                
                # Extract hand region for Unvoiced model
                hand_region = self.extract_hand_region_unvoiced(frame, hand_landmarks.landmark)
                
                if hand_region is not None:
                    # Predict using Unvoiced model
                    gesture, conf = self.unvoiced.predict_unvoiced(hand_region)
                    if gesture and conf > 0.5:  # Minimum confidence threshold
                        detected_gesture = gesture
                        confidence = conf
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
        if gesture and gesture in self.unvoiced.labels:
            self.current_text += gesture
            
            # Speak the gesture if TTS is enabled
            if self.tts_enabled and self.speak_on_detection:
                self.unvoiced.text_to_speech(gesture)
        
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
        print("Starting Unvoiced-Enhanced Sign Language Detection...")
        print("Detection running in background - check GUI for results")
        
        while self.detection_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect sign language using Unvoiced model
            frame, gesture, confidence = self.detect_sign_language_unvoiced(frame)
            
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
    
    def speak_text(self, text):
        """Speak the current text"""
        if self.tts_enabled and text:
            self.unvoiced.text_to_speech(text)
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class UnvoicedEnhancedGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Unvoiced-Enhanced Sign Language Translator")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2c3e50')
        
        # Create detector instance
        self.detector = UnvoicedEnhancedDetector()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_gui(self):
        """Setup the enhanced GUI elements"""
        # Title
        title_label = Label(self.root, text="Unvoiced-Enhanced Sign Language Translator", 
                          font=("Arial", 22, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        
        # Subtitle with Unvoiced credit
        subtitle_label = Label(self.root, text="Powered by Unvoiced (Inception V3) + MediaPipe", 
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
        
        model_info = self.detector.unvoiced.get_model_info()
        info_text = f"Model: {model_info['model_type']}\nClasses: {model_info['classes']}\nTTS: {'Enabled' if model_info['tts_enabled'] else 'Disabled'}"
        
        model_label = Label(model_info_frame, text=info_text, bg='#34495e', fg='white', 
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
            self.detector.speak_text(text)
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
            
            status_text += f"Model: Unvoiced (Inception V3)\n"
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
    
    print("Starting Unvoiced-Enhanced Sign Language Translator...")
    print("This version uses the pre-trained model from the Unvoiced repository")
    print("https://github.com/grassknoted/Unvoiced")
    
    # Start the GUI application
    app = UnvoicedEnhancedGUI()
    app.run()
