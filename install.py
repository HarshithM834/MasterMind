#!/usr/bin/env python3
"""
Sign Language Translator Installation Script
This script handles the complete setup and installation of the sign language translator.
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 70)
    print("🤟 SIGN LANGUAGE TRANSLATOR - INSTALLATION SCRIPT")
    print("=" * 70)
    print("This script will install and set up the Sign Language Translator")
    print("for real-time ASL gesture recognition using your webcam.")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected.")
        print("   This application requires Python 3.7 or higher.")
        print("   Please upgrade Python and try again.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
        return True

def check_system_info():
    """Display system information"""
    print(f"\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")

def install_pip_packages():
    """Install required pip packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.7", 
        "numpy==1.24.3",
        "tensorflow==2.13.0",
        "Pillow==10.0.1"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    modules = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("tkinter", "Tkinter")
    ]
    
    failed_imports = []
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"   ✅ {name} imported successfully")
        except ImportError as e:
            print(f"   ❌ Failed to import {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True

def test_webcam():
    """Test webcam functionality"""
    print("\n📹 Testing webcam...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ Could not open webcam")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print("   ✅ Webcam is working properly!")
            return True
        else:
            print("   ❌ Webcam detected but not working properly")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing webcam: {e}")
        return False

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if platform.system() == "Windows":
        print("\n🔗 Creating desktop shortcut...")
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Sign Language Translator.lnk")
            target = os.path.join(os.getcwd(), "run_sign_translator.py")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target}"'
            shortcut.WorkingDirectory = os.getcwd()
            shortcut.IconLocation = sys.executable
            shortcut.save()
            
            print("   ✅ Desktop shortcut created!")
            return True
        except ImportError:
            print("   ⚠️  Could not create desktop shortcut (winshell not available)")
            return False
        except Exception as e:
            print(f"   ❌ Error creating shortcut: {e}")
            return False
    else:
        print("   ⚠️  Desktop shortcut creation only supported on Windows")
        return False

def run_final_test():
    """Run a final test of the application"""
    print("\n🚀 Running final test...")
    
    try:
        # Test the simple demo
        print("   Testing simple demo...")
        result = subprocess.run([
            sys.executable, "-c", 
            "import cv2, mediapipe; print('✅ Core functionality working')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("   ✅ Final test passed!")
            return True
        else:
            print(f"   ❌ Final test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Test timed out")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 70)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 70)
    print("\n📖 How to use the Sign Language Translator:")
    print("\n1. 🚀 Quick Start:")
    print("   python run_sign_translator.py")
    print("\n2. 🎯 Enhanced Version (Recommended):")
    print("   python enhanced_sign_detector.py")
    print("\n3. 🧪 Simple Demo:")
    print("   python demo.py")
    print("\n4. 📋 Basic Version:")
    print("   python sign_language_detector.py")
    print("\n💡 Tips for best results:")
    print("   • Ensure good lighting on your hands")
    print("   • Keep hands 1-2 feet from the camera")
    print("   • Make clear, distinct ASL gestures")
    print("   • Use a plain background when possible")
    print("\n🔧 Troubleshooting:")
    print("   • If webcam issues occur, check camera permissions")
    print("   • For poor recognition, improve lighting conditions")
    print("   • Restart the application if it becomes unresponsive")
    print("\n📚 For more information, see README.md")
    print("=" * 70)

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Display system info
    check_system_info()
    
    # Install packages
    if not install_pip_packages():
        print("\n❌ Installation failed at package installation step.")
        print("   Please check your internet connection and try again.")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Installation failed at import testing step.")
        print("   Please check the error messages above and try reinstalling packages.")
        return
    
    # Test webcam
    webcam_ok = test_webcam()
    if not webcam_ok:
        print("\n⚠️  Webcam test failed. The application may not work properly.")
        print("   Please check your camera connection and permissions.")
    
    # Create desktop shortcut (Windows only)
    create_desktop_shortcut()
    
    # Run final test
    if not run_final_test():
        print("\n⚠️  Final test failed, but installation may still be successful.")
        print("   Try running the application manually to verify.")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during installation: {e}")
        print("Please check the error and try again.")
