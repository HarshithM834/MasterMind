#!/usr/bin/env python3
"""
Sign Language Translator Launcher
This script provides an easy way to launch the sign language translator application.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'tensorflow', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'numpy':
                import numpy
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'tkinter':
                import tkinter
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install opencv-python mediapipe numpy tensorflow")
        return False

def check_webcam():
    """Check if webcam is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Webcam detected and working!")
                return True
            else:
                print("❌ Webcam detected but not working properly")
                return False
        else:
            print("❌ Could not access webcam")
            return False
    except Exception as e:
        print(f"❌ Error checking webcam: {e}")
        return False

def launch_application():
    """Launch the sign language translator"""
    print("\n🚀 Starting Sign Language Translator...")
    
    # Check if enhanced version exists
    if os.path.exists("enhanced_sign_detector.py"):
        print("Running Enhanced Version...")
        try:
            subprocess.run([sys.executable, "enhanced_sign_detector.py"])
        except KeyboardInterrupt:
            print("\n👋 Application closed by user")
        except Exception as e:
            print(f"❌ Error running enhanced version: {e}")
            print("Trying basic version...")
            try:
                subprocess.run([sys.executable, "sign_language_detector.py"])
            except Exception as e2:
                print(f"❌ Error running basic version: {e2}")
    elif os.path.exists("sign_language_detector.py"):
        print("Running Basic Version...")
        try:
            subprocess.run([sys.executable, "sign_language_detector.py"])
        except KeyboardInterrupt:
            print("\n👋 Application closed by user")
        except Exception as e:
            print(f"❌ Error running application: {e}")
    else:
        print("❌ No sign language detector files found!")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🤟 SIGN LANGUAGE TRANSLATOR LAUNCHER")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    required_files = ["requirements.txt", "sign_language_detector.py"]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you're in the correct directory.")
        return
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        response = input("Would you like to install them automatically? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                return
        else:
            print("Please install the missing packages manually and try again.")
            return
    else:
        print("✅ All dependencies are installed!")
    
    # Check webcam
    print("\n📹 Checking webcam...")
    if not check_webcam():
        print("⚠️  Webcam issues detected. The application may not work properly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            return
    
    # Launch application
    launch_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher closed by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
