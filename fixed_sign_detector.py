import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk, scrolledtext
import queue
import os
import sys

class FixedSignLanguageDetector:
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enhanced ASL gesture mappings
        self.sign_mappings = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 
            'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 
            'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 
            'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            'SPACE': ' ', 'DELETE': 'DELETE', 'ENTER': 'ENTER'
        }
        
        # Gesture recognition parameters
        self.gesture_buffer = deque(maxlen=15)
        self.confidence_threshold = 0.75
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
        
    def enhanced_gesture_classification(self, landmarks):
        """Enhanced gesture classification with more sophisticated rules"""
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
                if (landmarks[tips[i]].y < landmarks[pip[i]].y and 
                    landmarks[pip[i]].y < landmarks[mcp[i]].y):
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
        
        extended_count = sum(fingers_extended)
        
        # Enhanced gesture recognition
        gesture = self.classify_enhanced_gesture(fingers_extended, extended_count, landmarks)
        
        return gesture
    
    def classify_enhanced_gesture(self, fingers_extended, extended_count, landmarks):
        """Enhanced gesture classification logic"""
        
        # A - Fist (no fingers extended)
        if extended_count == 0:
            return 'A'
        
        # B - All fingers extended except thumb
        elif extended_count == 4 and fingers_extended[0] == 0:
            return 'B'
        
        # C - Index and middle finger extended
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'C'
        
        # D - Only index finger extended
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'D'
        
        # E - All fingers extended
        elif extended_count == 5:
            return 'E'
        
        # F - Thumb and index finger extended
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'F'
        
        # G - Index finger extended, pointing
        elif extended_count == 1 and fingers_extended[1] == 1:
            # Check if pointing (index finger straight)
            if (landmarks[8].y < landmarks[6].y < landmarks[5].y):
                return 'G'
        
        # H - Index and middle finger extended, close together
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            # Check if fingers are close together
            tip_distance = abs(landmarks[8].x - landmarks[12].x)
            if tip_distance < 0.05:  # Distance between index and middle
                return 'H'
        
        # I - Pinky extended
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'I'
        
        # J - Pinky extended with hook motion (simplified)
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'J'
        
        # K - Index and middle finger extended, apart
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            tip_distance = abs(landmarks[8].x - landmarks[12].x)
            if tip_distance > 0.05:
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
            # Check if fingers are crossed
            if landmarks[8].x > landmarks[12].x:
                return 'R'
        
        # S - Fist with thumb over fingers
        elif extended_count == 0:
            if landmarks[4].y < landmarks[8].y:
                return 'S'
        
        # T - Thumb between index and middle fingers
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            if landmarks[4].x > landmarks[8].x and landmarks[4].x < landmarks[12].x:
                return 'T'
        
        # U - Index and middle finger extended, apart
        elif (extended_count == 2 and fingers_extended[1] == 1 and 
              fingers_extended[2] == 1):
            return 'U'
        
        # V - Index and middle finger extended, apart
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
    
    def detect_sign_language(self, frame):
        """Main detection function with enhanced processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with enhanced styling
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Classify gesture with enhanced method
                gesture = self.enhanced_gesture_classification(hand_landmarks.landmark)
                if gesture:
                    detected_gesture = gesture
                    confidence = 0.9  # High confidence for rule-based detection
                    self.gesture_buffer.append((gesture, confidence))
        
        return frame, detected_gesture, confidence
    
    def process_gesture_buffer(self):
        """Process gesture buffer with stability checking"""
        if len(self.gesture_buffer) < self.stability_frames:
            return None
        
        # Get most recent gestures
        recent_gestures = [gesture for gesture, _ in list(self.gesture_buffer)[-self.stability_frames:]]
        
        # Count gesture occurrences
        gesture_counts = {}
        for gesture in recent_gestures:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        if not gesture_counts:
            return None
        
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
                return most_common_gesture
        
        return None
    
    def update_text(self, gesture):
        """Update the current text based on detected gesture"""
        if gesture == 'SPACE':
            self.current_text += " "
        elif gesture == 'DELETE' and len(self.current_text) > 0:
            self.current_text = self.current_text[:-1]
        elif gesture == 'ENTER':
            self.current_text += "\n"
        elif gesture and gesture in self.sign_mappings:
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
        """Main detection loop - runs without OpenCV GUI"""
        print("Starting Enhanced Sign Language Detection...")
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
                final_gesture = self.process_gesture_buffer()
                if final_gesture:
                    self.update_text(final_gesture)
            
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
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class FixedSignLanguageGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Fixed Sign Language Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Create detector instance
        self.detector = FixedSignLanguageDetector()
        
        # Create GUI elements
        self.setup_gui()
        
        # Start update loop
        self.update_gui()
        
    def setup_gui(self):
        """Setup the enhanced GUI elements"""
        # Title
        title_label = Label(self.root, text="Fixed Sign Language Translator", 
                          font=("Arial", 24, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Status and controls
        left_panel = Frame(main_frame, bg='#34495e', relief=RAISED, bd=2)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
        
        # Status display area
        status_frame = Frame(left_panel, bg='black', height=400)
        status_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.status_label = Label(status_frame, text="Click 'Start Detection' to begin", 
                                bg="black", fg="white", font=("Arial", 14))
        self.status_label.pack(expand=True)
        
        # Control buttons
        button_frame = Frame(left_panel, bg='#34495e')
        button_frame.pack(fill=X, padx=10, pady=10)
        
        self.start_button = Button(button_frame, text="Start Detection", 
                                 command=self.start_detection, bg="#27ae60", fg="white",
                                 font=("Arial", 12, "bold"), padx=20, pady=10)
        self.start_button.pack(side=LEFT, padx=5)
        
        self.stop_button = Button(button_frame, text="Stop Detection", 
                                command=self.stop_detection, bg="#e74c3c", fg="white",
                                font=("Arial", 12, "bold"), padx=20, pady=10)
        self.stop_button.pack(side=LEFT, padx=5)
        
        self.clear_button = Button(button_frame, text="Clear Text", 
                                 command=self.clear_text, bg="#f39c12", fg="white",
                                 font=("Arial", 12, "bold"), padx=20, pady=10)
        self.clear_button.pack(side=LEFT, padx=5)
        
        self.quit_button = Button(button_frame, text="Quit", 
                                command=self.quit_app, bg="#95a5a6", fg="white",
                                font=("Arial", 12, "bold"), padx=20, pady=10)
        self.quit_button.pack(side=LEFT, padx=5)
        
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
                                                        height=20, width=40, 
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
                           text="Instructions: Start detection → Show ASL signs to camera → View translated text → Use controls as needed",
                           font=("Arial", 11), bg='#2c3e50', fg='#bdc3c7', justify=CENTER)
        instructions.pack()
        
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
                status_text += f"Confidence: {self.detector.current_confidence:.2f}\n"
            else:
                status_text += "No gesture detected\n"
            
            if hasattr(self.detector, 'current_fps'):
                status_text += f"FPS: {self.detector.current_fps}\n"
            
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
    
    print("Starting Fixed Sign Language Translator...")
    print("This version runs detection in background without OpenCV GUI")
    
    # Start the GUI application
    app = FixedSignLanguageGUI()
    app.run()
