from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.metrics import dp
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty, ObjectProperty
from kivy.gesture import Gesture, GestureDatabase

import cv2
import numpy as np
import time
import threading
import pytesseract
from ultralytics import YOLO
import pyttsx3
from PIL import Image as PILImage
import queue
import os as os

# Set the Tesseract path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\run59\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# For Android compatibility (if needed)
try:
    from android.permissions import request_permissions, Permission # type: ignore
    from android import mActivity # type: ignore
    android_imports = True
except ImportError:
    android_imports = False

class ButtonImage(ButtonBehavior, Image):
    """Custom button with image"""
    pass

class VisionAidLayout(FloatLayout):
    status_text = StringProperty("Starting up...")
    mode_text = StringProperty("Mode: Object & Hazard Detection")
    
    def __init__(self, **kwargs):
        # Initialize properties
        self.vision_system = None
        self.vision_thread = None
        
        # Call parent init which will trigger the kv rules
        super(VisionAidLayout, self).__init__(**kwargs)
        
        # Set up tap detection
        self._touch_time = time.time()
        self._touch_count = 0
        self._touch_timeout = 0.5
        
        # Initialize the application backend after UI is ready
        self.vision_system = VisionSystem(callback=self.update_ui)
        
        # Start the camera feed
        Clock.schedule_interval(self.update_camera, 1.0/30.0)  # 30 FPS
        
        # Start the vision system thread
        self.vision_thread = threading.Thread(target=self.vision_system.run)
        self.vision_thread.daemon = True
        self.vision_thread.start()
        
        # Set up accessibility mode (full screen tap detection)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
        # Request permissions for Android
        if android_imports:
            request_permissions([
                Permission.CAMERA,
                Permission.RECORD_AUDIO,
                Permission.WRITE_EXTERNAL_STORAGE
            ])
    
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'spacebar':
            # Simulate tap for testing on desktop
            self.handle_tap()
        return True
    
    def on_touch_down(self, touch):
        """Handle screen touches for tap counting"""
        if super(VisionAidLayout, self).on_touch_down(touch):
            return True
        
        current_time = time.time()
        
        # If touch is too old, reset counter
        if current_time - self._touch_time > self._touch_timeout:
            self._touch_count = 0
        
        self._touch_count += 1
        self._touch_time = current_time
        
        # Schedule the tap processing after timeout
        Clock.schedule_once(self.process_taps, self._touch_timeout)
        
        return True
    
    def handle_tap(self):
        """External method to handle taps (for testing)"""
        current_time = time.time()
        
        # If touch is too old, reset counter
        if current_time - self._touch_time > self._touch_timeout:
            self._touch_count = 0
        
        self._touch_count += 1
        self._touch_time = current_time
        
        # Schedule the tap processing after timeout
        Clock.schedule_once(self.process_taps, self._touch_timeout)
    
    def process_taps(self, dt):
        """Process tap sequence after timeout"""
        if self._touch_count == 0:
            return
        
        if 1 <= self._touch_count <= 3:
            # Change mode based on tap count
            mode_index = self._touch_count - 1
            mode_name = self.vision_system.set_mode(mode_index)
            self.mode_text = f"Mode: {mode_name}"
            self.status_text = f"Switched to {mode_name} mode"
        elif self._touch_count >= 4:
            # Special case: 4+ taps to take a snapshot for analysis in current mode
            self.vision_system.analyze_current_frame()
            self.status_text = "Taking snapshot for detailed analysis"
        
        self._touch_count = 0
    
    def update_camera(self, dt):
        """Update the camera feed in the UI"""
        frame = self.vision_system.get_current_frame()
        if frame is not None:
            # Convert to texture for Kivy
            buf = cv2.flip(frame, 0)  # Flip the buffer because Kivy's Y-axis is inverted
            buf = buf.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            # Update the image widget
            self.ids.camera_view.texture = texture
    
    def update_ui(self, status_text=None, mode_text=None):
        """Callback to update UI elements from the backend thread"""
        if status_text:
            self.status_text = status_text
        if mode_text:
            self.mode_text = mode_text
    
    def capture_scene(self):
        """Button handler for scene capture and analysis"""
        if self.vision_system and self.vision_system.current_mode == 1:  # Double-check we're in Scene mode
            self.vision_system.capture_and_analyze_scene()
    
    def capture_text(self):
        """Button handler for text capture and analysis"""
        if self.vision_system and self.vision_system.current_mode == 2:  # Double-check we're in Text mode
            self.vision_system.capture_and_analyze_text()

    def on_stop(self):
        """Clean up resources when the app is closing"""
        self.vision_system.stop()


class VisionSystem:
    """Backend vision processing system"""
    def __init__(self, callback=None):
        # Initialize running flag first
        self.running = True
        
        # Initialize YOLO models for different tasks
        self.object_model = YOLO("yolo11n.pt")  # General object recognition model
        self.scene_model =YOLO("yolo11s.pt")  # Will be loaded when mode is activated
        self.text_model = None   # Will be loaded when mode is activated
        
        # Path for scene model
        self.scene_model_path = "../yolo11s.pt"  # Path to scene description model
        
        # Lazy loading flags
        self.scene_model_loaded = False
        # Not using a model for text recognition since we're using Tesseract OCR directly
        
        # Store last captured text
        self.last_captured_text = ""
        
        # Flag to capture and analyze
        self.capture_requested = False
        self.text_capture_requested = False
        self.scene_capture_requested = False
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speed of speech
        
        # Voice queue to manage speech output
        self.voice_queue = queue.Queue()
        self.voice_thread = threading.Thread(target=self._process_voice_queue, daemon=True)
        self.voice_thread.start()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
        # Application modes - reduced to 3 modes (combined object & hazard)
        self.modes = ["Object & Hazard Detection", "Scene Description", "Text Recognition"]
        self.current_mode = 0
        
        # Safety alert thresholds (in pixels)
        self.proximity_threshold = 100  # Object proximity threshold
        self.hazard_classes = ['knife', 'scissors', 'fire', 'car', 'truck', 'motorcycle', 'person']
        
        # Scene description context enhancements
        self.scene_context = {
            'indoor': ['chair', 'table', 'couch', 'bed', 'tv', 'laptop', 'book', 'bottle'],
            'outdoor': ['tree', 'car', 'person', 'building', 'bicycle', 'traffic light', 'bird'],
            'kitchen': ['cup', 'bowl', 'spoon', 'fork', 'knife', 'oven', 'microwave', 'refrigerator'],
            'office': ['laptop', 'keyboard', 'mouse', 'desk', 'chair', 'monitor', 'phone'],
            'living_room': ['couch', 'tv', 'chair', 'table', 'clock', 'lamp', 'remote'],
            'bathroom': ['sink', 'toilet', 'toothbrush'],
            'street': ['car', 'truck', 'person', 'traffic light', 'stop sign', 'bicycle', 'motorcycle']
        }
        
        # Flag for taking snapshot for detailed analysis
        self.analyze_frame = False
        
        # Last spoken time for safety alerts to prevent constant repetition
        self.last_spoken = {key: 0 for key in self.hazard_classes}
        self.cooldown = 3  # seconds between repeated warnings
        
        # Last time path clear was announced
        self.last_path_clear_time = 0
        self.path_clear_cooldown = 5  # seconds between "path is clear" messages
        
        # Frame processing
        self.current_frame = None
        self.processed_frame = None
        self.frame_message = ""
        
        # UI callback
        self.callback = callback
        
        # Initial notification
        self.speak(f"VisionAid started in {self.modes[self.current_mode]} mode")
        
        if self.callback:
            self.callback(status_text="System initialized", 
                         mode_text=f"Mode: {self.modes[self.current_mode]}")
    
    def _load_scene_model(self):
        """Load the scene description model if not already loaded"""
        if not self.scene_model_loaded:
            try:
                self.speak("Loading scene description model")
                self.scene_model = YOLO(self.scene_model_path)
                self.scene_model_loaded = True
                self.speak("Scene description model loaded")
            except Exception as e:
                self.speak(f"Error loading scene model: {str(e)}")
                # Fallback to object model
                self.scene_model = self.object_model
    
    def _process_voice_queue(self):
        """Process voice messages in background thread"""
        # Store a local copy of the running flag for thread safety
        running_local = True
        
        while running_local:
            try:
                message = self.voice_queue.get(timeout=1)
                self.engine.say(message)
                self.engine.runAndWait()
                self.voice_queue.task_done()
            except queue.Empty:
                # Thread-safe check of the running flag
                try:
                    running_local = self.running
                except:
                    # If we can't access self.running, assume we should exit
                    break
                continue
            except Exception as e:
                print(f"Voice queue error: {str(e)}")
                time.sleep(1)  # Prevent busy-waiting if repeated errors
        
        print("Voice queue thread shutting down")

    def speak(self, text):
        """Add text to speech queue"""
        try:
            self.voice_queue.put(text)
            if self.callback:
                self.callback(status_text=text)
        except:
            print(f"Error adding text to voice queue: {text}")

    def set_mode(self, mode_index):
        """Change operating mode"""
        if 0 <= mode_index < len(self.modes):
            self.current_mode = mode_index
            mode_name = self.modes[self.current_mode]
            
            # Load appropriate model for the selected mode
            if mode_index == 1 and not self.scene_model_loaded:  # Scene Description
                threading.Thread(target=self._load_scene_model, daemon=True).start()
            
            self.speak(f"Switching to {mode_name} mode")
            return mode_name
        return self.modes[self.current_mode]

    def get_current_frame(self):
        """Get the most recent processed frame"""
        return self.processed_frame

    def analyze_current_frame(self):
        """Flag to perform detailed analysis on next frame"""
        self.analyze_frame = True
    
    def capture_and_analyze_scene(self):
        """Flag to capture and analyze scene in scene description mode"""
        if self.current_mode == 1:  # Scene Description mode
            self.scene_capture_requested = True
    
    def capture_and_analyze_text(self):
        """Flag to capture and analyze text in text recognition mode"""
        if self.current_mode == 2:  # Text Recognition mode
            self.text_capture_requested = True

    def stop(self):
        """Stop all threads and release resources"""
        print("Stopping vision system...")
        self.running = False
        
        # Ensure the voice queue is not left waiting
        try:
            # Add a dummy item to make the queue exit its blocking state
            self.voice_queue.put("Shutting down")
        except:
            pass
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        print("Vision system stopped")

    def object_and_hazard_detection(self, frame):
        """Combined object recognition and hazard detection mode"""
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        
        results = self.object_model(frame, stream=True, verbose=False)
        
        objects = {}
        alerts = []
        current_time = time.time()
        path_is_clear = True
        
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                if current_time - self.last_path_clear_time > self.path_clear_cooldown:
                    self.speak("Path is clear, no objects detected")
                    self.last_path_clear_time = current_time
                return frame, "No objects detected"
            
            # Process all detected objects
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                conf = float(box.conf[0])
                
                # Calculate box center
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                
                # Calculate distance from center
                distance_from_center = np.sqrt((box_center_x - center_x)**2 + 
                                              (box_center_y - center_y)**2)
                
                # Count objects
                if label in objects:
                    objects[label] += 1
                else:
                    objects[label] = 1
                
                # Default color: green for safe objects
                color = (0, 255, 0)
                
                # Check for hazards
                if label in self.hazard_classes:
                    # Draw warning for hazardous objects
                    color = (0, 165, 255)  # Orange for hazards
                    
                    # Check if close to center (user might be approaching)
                    if distance_from_center < self.proximity_threshold:
                        color = (0, 0, 255)  # Red for close hazards
                        path_is_clear = False
                        
                        # Prevent alert spamming with cooldown
                        if current_time - self.last_spoken.get(label, 0) > self.cooldown:
                            alert_msg = f"Warning! {label} nearby"
                            alerts.append(alert_msg)
                            self.speak(alert_msg)
                            self.last_spoken[label] = current_time
                
                # Draw rectangle with appropriate color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw distance indicator for potentially hazardous objects
                if label in self.hazard_classes:
                    cv2.line(frame, (center_x, center_y), (box_center_x, box_center_y), color, 1)
            
            # Find the most prominent object (largest bounding box) for main object reporting
            max_area = 0
            main_object = None
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    main_object = label
        
        # Announce path clear with cooldown to prevent constant repetition
        if path_is_clear and current_time - self.last_path_clear_time > self.path_clear_cooldown:
            self.speak("Path is clear")
            self.last_path_clear_time = current_time
        
        # Construct message for display
        if alerts:
            message = "; ".join(alerts)
        else:
            message = f"Main object: {main_object}" if main_object else "Processing frame"
        
        return frame, message

    def determine_scene_context(self, objects_dict):
        """Determine the context of the scene based on detected objects"""
        context_scores = {context: 0 for context in self.scene_context}
        
        # Count objects in each context category
        for obj in objects_dict.keys():
            for context, objects in self.scene_context.items():
                if obj in objects:
                    context_scores[context] += objects_dict[obj]
        
        # Find the most likely context
        likely_context = max(context_scores.items(), key=lambda x: x[1])
        
        # Only return a context if there's a minimum score
        if likely_context[1] >= 2:
            return likely_context[0]
        return None

    def scene_description(self, frame):
        """Generate a description of the whole scene with contextual understanding"""
        # Check if a capture was requested
        if self.scene_capture_requested:
            self.scene_capture_requested = False
            description = self.detailed_scene_description(frame)
            self.speak(description)
            return frame, description
        
        # Show a button overlay for scene capture
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw a rectangle in the center for the button
        btn_width, btn_height = int(w * 0.5), int(h * 0.15)
        btn_x1 = center_x - btn_width // 2
        btn_y1 = h - btn_height - 20  # Position at bottom of screen
        btn_x2 = center_x + btn_width // 2
        btn_y2 = h - 20
        
        # Draw the button
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 120, 255), -1)  # Filled button
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 0), 2)  # Border
        
        # Add text to button
        cv2.putText(frame, "Capture Scene", (btn_x1 + 20, btn_y1 + btn_height//2 + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw guide text at top
        guide_text = "Tap button to analyze scene"
        cv2.putText(frame, guide_text, (center_x - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, "Tap the button to capture and describe scene"

    def text_recognition(self, frame):
        """Show camera feed for text capture mode with a button for OCR"""
        # Check if a capture was requested
        if self.text_capture_requested:
            self.text_capture_requested = False
            result = self.capture_and_process_text(frame)
            return frame, result
        
        # Draw a target/focus area in the center of the frame to help users aim
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw a rectangle in the center to indicate the optimal text capture area
        rect_width, rect_height = int(w * 0.6), int(h * 0.3)
        rect_x1 = center_x - rect_width // 2
        rect_y1 = center_y - rect_height // 2
        rect_x2 = center_x + rect_width // 2
        rect_y2 = center_y + rect_height // 2
        
        # Draw the rectangle with a green color
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
        
        # Draw a button for text capture
        btn_width, btn_height = int(w * 0.5), int(h * 0.15)
        btn_x1 = center_x - btn_width // 2
        btn_y1 = h - btn_height - 20  # Position at bottom of screen
        btn_x2 = center_x + btn_width // 2
        btn_y2 = h - 20
        
        # Draw the button
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 120, 255), -1)  # Filled button
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 0), 2)  # Border
        
        # Add text to button
        cv2.putText(frame, "Capture Text", (btn_x1 + 20, btn_y1 + btn_height//2 + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add a guiding text
        guiding_message = "Aim at text and tap button to read"
        cv2.putText(frame, guiding_message, (center_x - 190, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check if we have a recently captured text to display
        if hasattr(self, 'last_captured_text') and self.last_captured_text:
            return frame, f"Last text: {self.last_captured_text[:100]}" + ("..." if len(self.last_captured_text) > 100 else "")
        
        return frame, guiding_message
    
    def capture_and_process_text(self, frame):
        """Process text when user explicitly requests through a tap"""
        self.speak("Analyzing text, please hold still")
        
        # Convert frame to grayscale for better OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply more extensive preprocessing to improve OCR accuracy
        # Resize to improve OCR performance
        scale_percent = 150  # Increase size by 150%
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Apply noise reduction
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding 
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Use pytesseract to detect text with custom configuration
        try:
            # Custom configuration for better text recognition
            custom_config = r'--oem 3 --psm 6'  # Assume single uniform block of text
            text = pytesseract.image_to_string(PILImage.fromarray(gray), config=custom_config)
            text = text.strip()
            
            if text:
                self.last_captured_text = text
                self.speak(f"Text found: {text}")
                return f"Text found: {text[:100]}" + ("..." if len(text) > 100 else "")
            else:
                # Try again with different preprocessing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(PILImage.fromarray(gray), config=custom_config)
                text = text.strip()
                
                if text:
                    self.last_captured_text = text
                    self.speak(f"Text found: {text}")
                    return f"Text found: {text[:100]}" + ("..." if len(text) > 100 else "")
                else:
                    self.last_captured_text = ""
                    self.speak("No text detected")
                    return "No text detected"
        except Exception as e:
            self.last_captured_text = ""
            error_msg = f"OCR error: {str(e)}"
            print(f"Tesseract error: {str(e)}")  # Log error for debugging
            self.speak("Error processing text")
            return error_msg

    def detailed_analysis(self, frame):
        """Perform detailed analysis based on current mode"""
        if self.current_mode == 0:  # Object & Hazard Detection
            result = self.detailed_object_and_hazard_analysis(frame)
            self.speak(result)
        elif self.current_mode == 1:  # Scene Description
            description = self.detailed_scene_description(frame)
            self.speak(description)
        elif self.current_mode == 2:  # Text Recognition
            text = self.detailed_text_recognition(frame)
            self.speak(f"The text says: {text}")

    def detailed_object_and_hazard_analysis(self, frame):
        """Detailed analysis for combined object and hazard detection mode"""
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        
        results = self.object_model(frame, verbose=False)
        
        objects = {}
        hazards = []
        path_is_clear = True
        
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Calculate box center
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                
                # Calculate distance from center
                distance = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
                
                # Count objects
                if label in objects:
                    objects[label] += 1
                else:
                    objects[label] = 1
                
                # Check for hazards
                if label in self.hazard_classes:
                    if distance < self.proximity_threshold:
                        hazards.append(f"{label} very close in front of you")
                        path_is_clear = False
                    else:
                        # Calculate rough direction
                        direction_x = "right" if box_center_x > center_x else "left"
                        direction_y = "below" if box_center_y > center_y else "above"
                        
                        hazards.append(f"{label} to the {direction_x} and {direction_y}")
        
        # Format the objects output
        objects_message = ""
        for obj, count in objects.items():
            if count == 1:
                objects_message += f"1 {obj}, "
            else:
                objects_message += f"{count} {obj}s, "
        
        objects_message = objects_message[:-2] if objects_message else "No objects detected"
        
        # Combine with hazard information
        if path_is_clear and not hazards:
            return f"Objects detected: {objects_message}. Path appears clear, no hazards detected."
        elif path_is_clear:
            return f"Objects detected: {objects_message}. Path is clear but be aware of: {', '.join(hazards)}"
        else:
            return f"Warning! {', '.join(hazards)}. Other objects: {objects_message}"

    def detailed_scene_description(self, frame):
        """Detailed scene description with contextual understanding"""
        # Use the appropriate model
        model = self.scene_model if self.scene_model_loaded and self.scene_model else self.object_model
        results = model(frame, verbose=False)
        
        objects = {}
        object_positions = {}
        
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                frame_h, frame_w = frame.shape[:2]
                
                # Determine object position in frame
                position_x = "left" if x1 < frame_w/3 else "right" if x2 > 2*frame_w/3 else "center"
                position_y = "top" if y1 < frame_h/3 else "bottom" if y2 > 2*frame_h/3 else "middle"
                position = f"{position_y} {position_x}"
                
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Store object counts for context determination
                if label in objects:
                    objects[label] += 1
                else:
                    objects[label] = 1
                
                # Store position information
                if position in object_positions:
                    object_positions[position].append(label)
                else:
                    object_positions[position] = [label]
        
        # Determine the context
        context = self.determine_scene_context(objects)
        
        # Generate descriptive text
        if not objects:
            return "No objects detected in the scene"
        
        description = ""
        if context:
            context_name = context.replace('_', ' ')
            description = f"You appear to be in a {context_name} environment. "
        
        # Add objects with their spatial relationships
        description += "In the scene I can see: "
        for position, items in object_positions.items():
            unique_items = list(set(items))  # Remove duplicates
            description += f"In the {position}: {', '.join(unique_items)}. "
        
        # Add potential interactions or relationships between objects
        if len(objects) > 1:
            description += self._infer_object_relationships(objects)
        
        return description

    def _infer_object_relationships(self, objects):
        """Infer relationships between objects to enhance scene description"""
        relationships = ""
        
        # Check for common relationships
        if 'person' in objects and 'chair' in objects:
            relationships += "Someone may be sitting on a chair. "
        elif 'person' in objects and 'couch' in objects:
            relationships += "Someone may be sitting on a couch. "
        elif 'cup' in objects and 'table' in objects:
            relationships += "There's a cup on the table. "
        elif 'book' in objects and ('table' in objects or 'desk' in objects):
            relationships += "There's a book on the table or desk. "
        elif ('car' in objects or 'truck' in objects) and 'road' in objects:
            relationships += "There are vehicles on the road. "
            
        # Check for kitchen context
        kitchen_items = ['cup', 'bowl', 'spoon', 'fork', 'knife', 'oven', 'microwave', 'refrigerator']
        kitchen_count = sum(1 for item in kitchen_items if item in objects)
        if kitchen_count >= 2:
            relationships += "This appears to be a kitchen setting. "
            
        # Check for outdoor context
        outdoor_items = ['tree', 'car', 'building', 'bicycle', 'traffic light']
        outdoor_count = sum(1 for item in outdoor_items if item in objects)
        if outdoor_count >= 2:
            relationships += "This appears to be an outdoor setting. "
        
        return relationships

    def detailed_text_recognition(self, frame):
        """Detailed text recognition - now consistently uses the same text processing logic as capture_and_process_text"""
        # Use the same text processing logic as normal text capture to be consistent
        return self.capture_and_process_text(frame)

    def run(self):
        """Main processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            # Store the original frame
            self.current_frame = frame.copy()
            
            # Process frame based on current mode
            if self.current_mode == 0:  # Object & Hazard Detection (combined mode)
                frame, message = self.object_and_hazard_detection(frame)
            elif self.current_mode == 1:  # Scene Description
                frame, message = self.scene_description(frame)
            elif self.current_mode == 2:  # Text Recognition
                frame, message = self.text_recognition(frame)
            
            # Add mode and message to frame
            cv2.putText(frame, f"Mode: {self.modes[self.current_mode]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, message, (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update processed frame for UI
            self.processed_frame = frame
            self.frame_message = message
            
            # Check if detailed analysis is requested
            if self.analyze_frame:
                self.detailed_analysis(frame)
                self.analyze_frame = False
                
            # Update UI if callback provided
            if self.callback:
                self.callback(status_text=message, 
                             mode_text=f"Mode: {self.modes[self.current_mode]}")
            
            # Short delay to prevent high CPU usage
            time.sleep(0.01)


# Define the Kivy UI layout using Builder
from kivy.lang import Builder

kv_string = '''
<VisionAidLayout>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(10)
        spacing: dp(10)
        
        # Camera preview
        BoxLayout:
            size_hint_y: 0.7
            
            ButtonImage:
                id: camera_view
                on_press: root.handle_tap()
        
        # Action buttons for scene and text modes
        BoxLayout:
            size_hint_y: 0.1
            orientation: 'horizontal'
            spacing: dp(10)
            
            Button:
                id: scene_capture_button
                text: 'Capture Scene'
                opacity: 1 if root.mode_text == "Mode: Scene Description" else 0
                disabled: not (root.mode_text == "Mode: Scene Description")
                on_press: root.capture_scene()
            
            Button:
                id: text_capture_button
                text: 'Capture Text'
                opacity: 1 if root.mode_text == "Mode: Text Recognition" else 0
                disabled: not (root.mode_text == "Mode: Text Recognition")
                on_press: root.capture_text()
        
        # Status bar
        BoxLayout:
            size_hint_y: 0.2
            orientation: 'vertical'
            
            Label:
                id: mode_label
                text: root.mode_text
                font_size: dp(20)
                halign: 'center'
                
            Label:
                id: status_label
                text: root.status_text
                font_size: dp(16)
                halign: 'center'
'''

Builder.load_string(kv_string)

class VisionAidApp(App):
    def build(self):
        # Set window size for desktop testing
        if not android_imports:
            Window.size = (480, 800)  # Simulate phone dimensions
        
        return VisionAidLayout()
    
    def on_stop(self):
        # Make sure to clean up resources
        self.root.vision_system.stop()

if __name__ == "__main__":
    VisionAidApp().run()