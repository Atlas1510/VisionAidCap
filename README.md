# VisionAid: Computer Vision Assistant for the Visually Impaired

VisionAid is a mobile application built with Kivy and various computer vision technologies to assist visually impaired users by detecting objects, describing scenes, and recognizing text in their environment.

## Features

- **Object & Hazard Detection**: Identifies objects and potential hazards in the user's path
- **Scene Description**: Provides detailed descriptions of the user's surroundings
- **Text Recognition**: Reads text from documents, signs, and other sources
- **Accessible Interface**: Simple tap-based interaction for mode switching and functionality
- **Voice Feedback**: Provides audio descriptions of detected information

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- For Android: Android device with camera

### Required Python Packages

```
kivy
opencv-python (cv2)
numpy
pytesseract
ultralytics (YOLO)
pyttsx3
pillow (PIL)
```

### Installation Steps

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/visionaid.git
   cd visionaid
   ```

2. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```
   pip install kivy opencv-python numpy pytesseract ultralytics pyttsx3 pillow
   ```

4. **Install Tesseract OCR**:

   - **Windows**:
     1. Download the installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
     2. Run the installer and complete the installation
     3. Install for the user not the entire machine.
     4. Update the path in the code to match your installation location:
        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\---\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        ```

   - **macOS**:
     ```
     brew install tesseract
     ```

   - **Linux (Ubuntu/Debian)**:
     ```
     sudo apt-get install tesseract-ocr
     ```

5. **Download YOLO models**:
   - Download YOLO models from the Ultralytics repository
   - Rename them to match the code expectations (`yolo11n.pt` and `yolo11s.pt`)
   - Place them in the project directory

6. **Run the application**:
   ```
   python VisionaidTap_Final.py
   ```

### Theoretical Android Setup

Note: The following steps describe how you would theoretically set up the application for Android, but this implementation is not fully complete in the current code:

1. Install Buildozer:
   ```
   pip install buildozer
   ```

2. Initialize Buildozer in your project directory:
   ```
   buildozer init
   ```

3. Edit the buildozer.spec file to include all requirements and permissions:
   - Add python packages: kivy, opencv, numpy, pytesseract, ultralytics, pyttsx3, pillow
   - Add permissions: CAMERA, RECORD_AUDIO, WRITE_EXTERNAL_STORAGE

4. Build the APK (theoretical):
   ```
   buildozer android debug
   ```

5. The code includes Android permission handling, but would require additional modifications to be fully functional on Android devices

## Usage Instructions

- **Single Tap**: Switch to Object & Hazard Detection mode
- **Double Tap**: Switch to Scene Description mode
- **Triple Tap**: Switch to Text Recognition mode
- **Mode-Specific Buttons**: Use the on-screen buttons to trigger actions specific to each mode (scene capture in Scene Description mode, text capture in Text Recognition mode)

## Code Explanation

### Main Components

The application is structured around three main components:

1. **VisionAidApp**: The main Kivy application class
2. **VisionAidLayout**: The UI layout manager for the application
3. **VisionSystem**: The backend vision processing system

Let's explore each section of the code in detail:

### Imports and Setup

The application uses a variety of libraries for different purposes:

- **Kivy**: For the user interface and mobile application framework
- **OpenCV (cv2)**: For camera access and image processing
- **YOLO (Ultralytics)**: For object detection and scene understanding
- **Pytesseract**: For optical character recognition (text detection)
- **Pyttsx3**: For text-to-speech conversion
- **PIL (Pillow)**: For image handling compatible with Pytesseract
- **Threading and Queue**: For parallel processing to keep the UI responsive

### User Interface (VisionAidLayout)

The `VisionAidLayout` class:

- Manages the touch interface for tap detection (1-3 taps to switch modes)
- Handles the camera preview display
- Provides mode-specific buttons for actions like scene capture and text recognition
- Updates status text based on vision system feedback
- Translates user interaction to vision system commands

The interface is designed with accessibility in mind, using large buttons and a simple tap system to switch between modes. The tap system specifically counts 1-3 taps to switch between the three available modes.

### Vision Processing System (VisionSystem)

The `VisionSystem` class implements the core functionality:

#### Initialization and Setup

- Loads the appropriate YOLO models for object detection and scene description
- Sets up the text-to-speech engine
- Initializes the camera
- Creates a background thread for voice output to prevent UI freezing

#### Mode Management

The system has three primary modes:

1. **Object & Hazard Detection**:
   - Identifies objects in the camera view
   - Highlights potential hazards (items from the hazard class list)
   - Provides warnings about nearby hazards
   - Announces when the path is clear

2. **Scene Description**:
   - Analyzes the entire scene to understand the context (indoor, outdoor, kitchen, etc.)
   - Identifies objects and their spatial relationships
   - Creates natural language descriptions of the environment
   - Infers relationships between objects for more meaningful descriptions

3. **Text Recognition**:
   - Provides a target area to help users aim at text
   - Preprocesses images to improve OCR accuracy
   - Uses Tesseract to recognize text in images
   - Reads recognized text aloud

#### Video Processing Loop

The main processing loop:
- Captures frames from the camera
- Processes frames based on the current mode
- Updates the UI with processed frames and status messages
- Handles detailed analysis requests
- Manages the voice output queue

### Scene Context and Object Relationships

The system uses contextual understanding to provide more meaningful descriptions:

- Scene contexts like "indoor", "outdoor", "kitchen", etc. are determined by the objects present
- Object relationships are inferred based on common associations (e.g., person and chair suggests sitting)
- Spatial relationships are described using natural language (left, right, center, etc.)

### Safety Features

The system includes several safety-focused features:

- Hazard detection with proximity warnings
- Cooldown timers to prevent alert spamming
- Path clearance announcements
- Visual indicators for detected hazards (color-coded bounding boxes)

## Troubleshooting

### Common Issues

1. **Tesseract OCR Not Found**:
   - Verify Tesseract is installed correctly
   - Check the path in `pytesseract.pytesseract.tesseract_cmd`

2. **YOLO Models Not Loading**:
   - Ensure you've downloaded the correct models
   - Verify file paths are correct
   - Check console for specific error messages

3. **Camera Not Accessible**:
   - Ensure your webcam/camera is connected and working
   - Check if other applications are using the camera
   - Try changing the camera index (`cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

4. **Platform Compatibility**:
   - This application is primarily designed for desktop/laptop use with a webcam
   - The code includes Android-specific imports, but the Android functionality is theoretical and would require additional implementation

## Extending the Application

To extend the application with new features:

1. **New Detection Models**: Replace or add new YOLO models for specialized detection tasks
2. **Additional Modes**: Add new modes by creating new processing methods in `VisionSystem`
3. **UI Enhancements**: Modify the Kivy layout in the `kv_string` variable
4. **Gesture Recognition**: Expand the tap system to include swipes or other gestures

## Acknowledgments

- Ultralytics for the YOLO object detection models
- Tesseract OCR for text recognition capabilities
- The Kivy framework for cross-platform mobile development
