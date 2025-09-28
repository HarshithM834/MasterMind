#!/usr/bin/env python3
"""
Simple Demo Script for Sign Language Detection
This script provides a minimal example of how to use the sign language detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class SimpleSignDetector:
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
        
        # Simple gesture buffer
        self.gesture_buffer = deque(maxlen=10)
        self.current_text = ""
        
    def classify_simple_gesture(self, landmarks):
        """Simple gesture classification"""
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
        
        extended_count = sum(fingers_extended)
        
        # Simple classification
        if extended_count == 0:
            return 'A'  # Fist
        elif extended_count == 1 and fingers_extended[1] == 1:
            return 'B'  # Index finger
        elif extended_count == 2 and fingers_extended[1] == 1 and fingers_extended[2] == 1:
            return 'C'  # Index and middle
        elif extended_count == 4 and fingers_extended[0] == 0:
            return 'D'  # All fingers except thumb
        elif extended_count == 5:
            return 'E'  # All fingers
        else:
            return None
    
    def run_demo(self):
        """Run the simple demo"""
        print("Simple Sign Language Detection Demo")
        print("Press 'q' to quit, 'c' to clear text")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            detected_gesture = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Classify gesture
                    gesture = self.classify_simple_gesture(hand_landmarks.landmark)
                    if gesture:
                        detected_gesture = gesture
                        self.gesture_buffer.append(gesture)
            
            # Process gesture buffer
            if len(self.gesture_buffer) >= 5:
                gesture_counts = {}
                for gesture in self.gesture_buffer:
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                
                most_common_gesture = max(gesture_counts, key=gesture_counts.get)
                confidence = gesture_counts[most_common_gesture] / len(self.gesture_buffer)
                
                if confidence >= 0.7:  # 70% confidence threshold
                    if most_common_gesture != getattr(self, 'last_gesture', None):
                        self.last_gesture = most_common_gesture
                        self.current_text += most_common_gesture
                        print(f"Detected: {most_common_gesture} | Text: {self.current_text}")
            
            # Display current text on frame
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {detected_gesture if detected_gesture else 'None'}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Simple Sign Language Demo', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
                print("Text cleared")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main demo function"""
    print("=" * 50)
    print("Simple Sign Language Detection Demo")
    print("=" * 50)
    
    # Check if webcam is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    cap.release()
    
    # Create and run detector
    detector = SimpleSignDetector()
    detector.run_demo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDemo closed by user")
    except Exception as e:
        print(f"Error running demo: {e}")
