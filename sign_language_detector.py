import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import threading
import time
from tkinter import *
from tkinter import ttk
import queue

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
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
        
        # Sign language mappings (basic ASL alphabet)
        self.sign_mappings = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'DELETE'
        }
        
        # Simple gesture recognition based on finger positions
        self.gesture_buffer = deque(maxlen=10)
        self.current_text = ""
        self.translation_queue = queue.Queue()
        
        # Create a simple neural network for gesture classification
        self.model = self.create_simple_model()
        
    def create_simple_model(self):
        """Create a simple neural network for gesture classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),  # 21 landmarks * 2 coordinates
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(28, activation='softmax')  # 26 letters + space + delete
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # For now, return untrained model (in production, you'd load a trained model)
        return model
    
    def extract_hand_features(self, landmarks):
        """Extract normalized hand landmark features"""
        if landmarks is None:
            return None
        
        # Normalize landmarks relative to wrist
        wrist = landmarks[0]
        normalized_landmarks = []
        
        for landmark in landmarks:
            normalized_landmarks.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y
            ])
        
        return np.array(normalized_landmarks)
    
    def classify_gesture(self, landmarks):
        """Classify gesture using simple rule-based approach"""
        if landmarks is None:
            return None
        
        # Get finger tip positions
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        mcp = [2, 5, 9, 13, 17]   # MCP joints
        
        # Check if fingers are extended
        fingers_extended = []
        for i in range(5):
            if i == 0:  # Thumb
                if landmarks[tips[i]].x > landmarks[mcp[i]].x:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
            else:  # Other fingers
                if landmarks[tips[i]].y < landmarks[mcp[i]].y:
                    fingers_extended.append(1)
                else:
                    fingers_extended.append(0)
        
        # Simple gesture classification based on finger positions
        extended_count = sum(fingers_extended)
        
        # Basic ASL alphabet gestures (simplified)
        if extended_count == 0:
            return 'A'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'B'
        elif extended_count == 1 and fingers_extended[0] == 1:
            return 'C'
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'D'
        elif extended_count == 4:
            return 'E'
        elif extended_count == 1 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'F'
        elif extended_count == 3 and fingers_extended[0] == 0:
            return 'G'
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[1] == 1:
            return 'H'
        elif extended_count == 1 and fingers_extended[2] == 1:
            return 'I'
        elif extended_count == 1 and fingers_extended[3] == 1:
            return 'J'
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[4] == 1:
            return 'K'
        elif extended_count == 1 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'L'
        elif extended_count == 3 and fingers_extended[0] == 1:
            return 'M'
        elif extended_count == 2 and fingers_extended[0] == 1 and fingers_extended[2] == 1:
            return 'N'
        elif extended_count == 0 and abs(landmarks[8].y - landmarks[12].y) < 0.05:
            return 'O'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'P'
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'Q'
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'R'
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[4] == 1:
            return 'S'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'T'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'U'
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'V'
        elif extended_count == 3 and fingers_extended[1] == 1 and fingers_extended[2] == 1 and fingers_extended[3] == 1:
            return 'W'
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'X'
        elif extended_count == 1 and fingers_extended[4] == 1:
            return 'Y'
        elif extended_count == 0:
            return 'Z'
        else:
            return None
    
    def detect_sign_language(self, frame):
        """Main detection function"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Classify gesture
                gesture = self.classify_gesture(hand_landmarks.landmark)
                if gesture:
                    detected_gesture = gesture
                    # Add gesture to buffer
                    self.gesture_buffer.append(gesture)
        
        return frame, detected_gesture
    
    def process_gesture_buffer(self):
        """Process gesture buffer to determine final gesture"""
        if len(self.gesture_buffer) < 5:
            return None
        
        # Get most common gesture in buffer
        gesture_counts = {}
        for gesture in self.gesture_buffer:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        confidence = gesture_counts[most_common_gesture] / len(self.gesture_buffer)
        
        if confidence >= 0.7:  # 70% confidence threshold
            return most_common_gesture
        return None
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting sign language detection...")
        print("Press 'q' to quit, 'c' to clear text, 's' to add space")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect sign language
            frame, gesture = self.detect_sign_language(frame)
            
            # Process gesture if detected
            if gesture:
                final_gesture = self.process_gesture_buffer()
                if final_gesture and final_gesture != self.last_gesture:
                    self.last_gesture = final_gesture
                    self.update_text(final_gesture)
            
            # Display current text on frame
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Sign Language Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
            elif key == ord('s'):
                self.current_text += " "
        
        self.cleanup()
    
    def update_text(self, gesture):
        """Update the current text based on detected gesture"""
        if gesture == 'SPACE':
            self.current_text += " "
        elif gesture == 'DELETE' and len(self.current_text) > 0:
            self.current_text = self.current_text[:-1]
        else:
            self.current_text += gesture
        
        print(f"Current text: {self.current_text}")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

class SignLanguageGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Sign Language Translator")
        self.root.geometry("800x600")
        
        # Create detector instance
        self.detector = SignLanguageDetector()
        
        # Create GUI elements
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI elements"""
        # Title
        title_label = Label(self.root, text="Sign Language Translator", 
                          font=("Arial", 20, "bold"))
        title_label.pack(pady=20)
        
        # Video display area
        self.video_label = Label(self.root, text="Video Feed Will Appear Here", 
                               bg="black", fg="white", width=80, height=20)
        self.video_label.pack(pady=10)
        
        # Translated text display
        self.text_label = Label(self.root, text="Translated Text:", 
                              font=("Arial", 14, "bold"))
        self.text_label.pack(pady=5)
        
        self.translated_text = Text(self.root, height=8, width=80, 
                                  font=("Arial", 12))
        self.translated_text.pack(pady=10, padx=20, fill=BOTH, expand=True)
        
        # Control buttons
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        start_button = Button(button_frame, text="Start Detection", 
                            command=self.start_detection, bg="green", fg="white")
        start_button.pack(side=LEFT, padx=5)
        
        clear_button = Button(button_frame, text="Clear Text", 
                            command=self.clear_text, bg="red", fg="white")
        clear_button.pack(side=LEFT, padx=5)
        
        quit_button = Button(button_frame, text="Quit", 
                           command=self.quit_app, bg="gray", fg="white")
        quit_button.pack(side=LEFT, padx=5)
        
        # Instructions
        instructions = Label(self.root, 
                           text="Instructions:\n"
                                "1. Click 'Start Detection' to begin\n"
                                "2. Show ASL signs to the camera\n"
                                "3. Translated text will appear below\n"
                                "4. Use 'Clear Text' to reset\n"
                                "5. Press 'q' in video window to stop detection",
                           font=("Arial", 10), justify=LEFT)
        instructions.pack(pady=10)
        
    def start_detection(self):
        """Start the detection process"""
        # Run detection in a separate thread
        detection_thread = threading.Thread(target=self.detector.run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
    def clear_text(self):
        """Clear the translated text"""
        self.translated_text.delete(1.0, END)
        self.detector.current_text = ""
        
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
    
    # Start the GUI application
    app = SignLanguageGUI()
    app.run()
