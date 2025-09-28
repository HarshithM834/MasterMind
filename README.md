# Sign Language Translator

A real-time sign language detection and translation application that uses your webcam to detect American Sign Language (ASL) gestures and converts them to text.

## Features

- **Real-time Detection**: Live webcam feed with hand tracking
- **ASL Alphabet Support**: Recognizes all 26 letters of the ASL alphabet
- **Enhanced GUI**: Modern, user-friendly interface with real-time text display
- **Gesture Stability**: Advanced filtering to ensure accurate gesture recognition
- **Performance Monitoring**: FPS counter and confidence metrics
- **Multiple Controls**: Keyboard shortcuts and GUI buttons for easy operation

## Requirements

- Python 3.7 or higher
- Webcam/camera
- OpenCV
- MediaPipe
- TensorFlow
- Tkinter (usually comes with Python)

## Installation

1. **Clone or download this repository**
   ```bash
   cd /path/to/your/project
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install opencv-python==4.8.1.78
   pip install mediapipe==0.10.7
   pip install numpy==1.24.3
   pip install tensorflow==2.13.0
   pip install Pillow==10.0.1
   ```

3. **Verify webcam access**
   Make sure your webcam is connected and working properly.

## Usage

### Basic Version
```bash
python sign_language_detector.py
```

### Enhanced Version (Recommended)
```bash
python enhanced_sign_detector.py
```

## How to Use

1. **Start the Application**
   - Run the enhanced version for the best experience
   - The GUI will open with a video feed area and text display

2. **Begin Detection**
   - Click "Start Detection" button
   - A new window will open showing your webcam feed
   - Position your hands in front of the camera

3. **Sign Language Recognition**
   - Show ASL signs clearly to the camera
   - Keep your hands well-lit and visible
   - The system will detect gestures and display them as text

4. **Controls**
   - **GUI Controls**:
     - "Start Detection": Begin gesture recognition
     - "Clear Text": Reset the translated text
     - "Quit": Exit the application
   
   - **Keyboard Controls** (in video window):
     - `q`: Quit detection
     - `c`: Clear text
     - `s`: Add space
     - `e`: Add new line

## Supported Gestures

The application recognizes the following ASL gestures:

- **A-Z**: All letters of the alphabet
- **Space**: Gesture for adding spaces
- **Delete**: Gesture for backspace/delete
- **Enter**: Gesture for new line

## Gesture Recognition Tips

1. **Lighting**: Ensure good lighting on your hands
2. **Distance**: Keep hands 1-2 feet from the camera
3. **Stability**: Hold gestures steady for 1-2 seconds
4. **Clarity**: Make clear, distinct hand shapes
5. **Background**: Use a plain background when possible

## Technical Details

### Architecture
- **Hand Detection**: Uses MediaPipe for robust hand landmark detection
- **Gesture Classification**: Rule-based classification with stability filtering
- **Real-time Processing**: Optimized for 30 FPS performance
- **GUI Framework**: Tkinter for cross-platform compatibility

### Performance Features
- **FPS Monitoring**: Real-time frame rate display
- **Confidence Scoring**: Gesture recognition confidence metrics
- **Stability Filtering**: Prevents false positives from shaky gestures
- **Multi-threading**: Separate threads for detection and GUI updates

### Future Enhancements
- Machine learning model training for improved accuracy
- Support for ASL words and phrases
- Voice synthesis for text-to-speech
- Multiple sign language support
- Mobile app version

## Troubleshooting

### Common Issues

1. **Webcam not detected**
   - Check camera permissions
   - Ensure webcam is not being used by another application
   - Try restarting the application

2. **Poor gesture recognition**
   - Improve lighting conditions
   - Adjust camera distance
   - Make clearer, more distinct gestures
   - Ensure hands are fully visible

3. **Performance issues**
   - Close other applications using the camera
   - Reduce video resolution if needed
   - Check system resources

4. **Installation errors**
   - Update pip: `pip install --upgrade pip`
   - Install packages individually if batch installation fails
   - Check Python version compatibility

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **Camera**: USB webcam or built-in camera
- **Python**: Version 3.7 or higher

## Contributing

This is an open-source project. Contributions are welcome! Areas for improvement:

- Enhanced gesture recognition algorithms
- Support for more sign languages
- Machine learning model improvements
- UI/UX enhancements
- Performance optimizations

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe team for hand detection capabilities
- OpenCV community for computer vision tools
- ASL community for sign language resources

---

**Note**: This application is designed for educational and accessibility purposes. For professional sign language interpretation, please consult certified interpreters.
