import sys
import cv2
import numpy as np
import threading
import time
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO
from PIL import Image

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
                               QProgressBar, QFrame, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

class VideoThread(QThread):
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.running = False
        
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_source)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                # Video ended, restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(1/30)  # 30 FPS
            
        cap.release()
        
    def stop(self):
        self.running = False
        self.wait()

class ProcessingThread(QThread):
    result_ready = Signal(np.ndarray, dict)
    progress_updated = Signal(int)
    
    def __init__(self, frame, yolo_model, birefnet_model, birefnet_transform, settings):
        super().__init__()
        self.frame = frame
        self.yolo_model = yolo_model
        self.birefnet_model = birefnet_model
        self.birefnet_transform = birefnet_transform
        self.settings = settings
        
    def run(self):
        try:
            self.progress_updated.emit(10)
            
            # YOLOv11 Detection
            confidence = self.settings['confidence']
            results = self.yolo_model(self.frame, conf=confidence)
            
            self.progress_updated.emit(30)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the first detection
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Crop the detected object
                cropped = self.frame[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    self.progress_updated.emit(50)
                    
                    # BirefNet Segmentation (if available)
                    if self.birefnet_model is not None:
                        try:
                            mask = self.run_birefnet_inference(cropped)
                        except Exception as e:
                            print(f"BirefNet inference failed: {e}")
                            # Create a simple rectangular mask
                            mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
                    else:
                        # Create a simple rectangular mask if BirefNet is not available
                        mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
                    
                    self.progress_updated.emit(70)
                    
                    # Process results
                    result_frame, measurements = self.process_results(self.frame, cropped, mask, x1, y1, x2, y2)
                    
                    self.progress_updated.emit(100)
                    self.result_ready.emit(result_frame, measurements)
                else:
                    self.result_ready.emit(self.frame.copy(), {'error': 'No valid object detected'})
            else:
                self.result_ready.emit(self.frame.copy(), {'error': 'No objects detected'})
                
        except Exception as e:
            self.result_ready.emit(self.frame.copy(), {'error': f'Processing failed: {str(e)}'})
    
    def run_birefnet_inference(self, image: np.ndarray, score_th: Optional[float] = 0.5) -> np.ndarray:
        """Run BirefNet inference on cropped image"""
        import time
        
        if self.birefnet_model is None or self.birefnet_transform is None:
            raise ValueError("BirefNet model not loaded")
            
        device = next(self.birefnet_model.parameters()).device
        
        # Pre process: Convert to PIL, apply transform
        preprocess_start = time.time()
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transform
        input_tensor = self.birefnet_transform(pil_image).unsqueeze(0).to(device)
        preprocess_time = time.time() - preprocess_start

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            output = self.birefnet_model(input_tensor)
            # Debug: Print output type and structure
            print(f"BirefNet output type: {type(output)}")
            if hasattr(output, '__len__'):
                print(f"BirefNet output length: {len(output)}")
            if hasattr(output, 'shape'):
                print(f"BirefNet output shape: {output.shape}")
                
            # Handle BirefNet output format: list with last element being the prediction
            if isinstance(output, (list, tuple)):
                # Take the last element and apply sigmoid
                preds = output[-1].sigmoid().cpu()
                mask = preds[0].squeeze()
                print(f"BirefNet prediction shape: {mask.shape}")
            else:
                # Fallback for other formats
                mask = output
        inference_time = time.time() - inference_start

        # Post process: Convert to numpy, resize to original size
        postprocess_start = time.time()
        
        # Convert to numpy (sigmoid already applied)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            # If it's already numpy array
            mask_np = np.array(mask)
            
        # Debug: Print mask info
        print(f"Mask type: {type(mask_np)}, shape: {mask_np.shape}, dtype: {mask_np.dtype}")
        print(f"Mask min: {mask_np.min()}, max: {mask_np.max()}")
            
        if len(mask_np.shape) > 2:
            mask_np = mask_np[0]  # Take first channel if multiple channels
            
        # Apply threshold (sigmoid already applied)
        if score_th is not None:
            mask_np = np.where(mask_np < score_th, 0, 1)
            
        # Ensure mask is valid for resize
        if mask_np.size == 0 or np.isnan(mask_np).any():
            print("Warning: Invalid mask detected, creating default mask")
            mask_np = np.ones((1024, 1024), dtype=np.float32)
            
        # Resize to original image size
        try:
            # Ensure mask is float32 and has correct dimensions
            mask_np = mask_np.astype(np.float32)
            if len(mask_np.shape) == 2:
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                print(f"Unexpected mask shape: {mask_np.shape}")
                mask_resized = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        except Exception as e:
            print(f"Resize error: {e}")
            print(f"Mask shape: {mask_np.shape}, target size: ({image.shape[1]}, {image.shape[0]})")
            # Fallback: create a simple mask
            mask_resized = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Convert to uint8
        mask_final = (mask_resized * 255).astype(np.uint8)
        postprocess_time = time.time() - postprocess_start

        # Print BirefNet detailed timing
        print(f"BirefNet Details:")
        print(f"  Input size: {image.shape[1]}x{image.shape[0]} -> 1024x1024")
        print(f"  Device: {device}")
        print(f"  Preprocessing: {preprocess_time:.3f}s ({preprocess_time*1000:.1f}ms)")
        print(f"  PyTorch Inference: {inference_time:.3f}s ({inference_time*1000:.1f}ms)")
        print(f"  Postprocessing: {postprocess_time:.3f}s ({postprocess_time*1000:.1f}ms)")

        return mask_final
    
    def process_results(self, frame, cropped, mask, x1, y1, x2, y2):
        """Process segmentation results and draw measurements"""
        # Create full frame mask
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask_resized = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]))
        full_mask[y1:y2, x1:x2] = mask_resized
        
        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_frame = frame.copy()
        measurements = {}
        
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                contour_threshold = self.settings['contour_threshold']
                
                if area > contour_threshold and i == 0:  # Only process largest contour
                    # Corner detection
                    epsilon_value = self.settings['epsilon']
                    epsilon = epsilon_value * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) < 3:
                        epsilon = epsilon_value * 0.5 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) < 3:
                            approx = contour
                    
                    if len(approx) >= 2:
                        # Draw corner points
                        corner_points = []
                        for j, point in enumerate(approx):
                            x, y = point.ravel()
                            corner_points.append((int(x), int(y)))
                            
                            # Draw larger corner points with better visibility
                            cv2.circle(result_frame, (int(x), int(y)), 15, (0, 0, 255), -1)  # Red fill
                            cv2.circle(result_frame, (int(x), int(y)), 15, (255, 255, 255), 3)  # White border
                            
                            # Add corner number with better positioning
                            corner_text = f'C{j+1}'
                            (text_width, text_height), _ = cv2.getTextSize(corner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                            
                            # Background for corner number
                            cv2.rectangle(result_frame, 
                                        (int(x) + 20, int(y) - text_height//2 - 5), 
                                        (int(x) + 20 + text_width + 10, int(y) + text_height//2 + 5), 
                                        (0, 0, 0), -1)
                            
                            # Corner number text
                            cv2.putText(result_frame, corner_text, 
                                      (int(x) + 25, int(y) + text_height//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        
                        # Calculate distances between corners (approximated)
                        corner_distances = []
                        corner_total_distance = 0
                        
                        for j in range(len(corner_points)):
                            p1 = corner_points[j]
                            p2 = corner_points[(j + 1) % len(corner_points)]
                            
                            # Calculate distance between corners
                            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            pixel_cm_ratio = self.settings['pixel_cm_ratio']
                            distance_cm = distance * pixel_cm_ratio
                            
                            corner_distances.append(distance_cm)
                            corner_total_distance += distance_cm
                        
                        # Calculate actual contour length using all contour points
                        actual_contour_length = 0
                        contour_points = contour.reshape(-1, 2)  # Reshape contour to points array
                        
                        for i in range(len(contour_points)):
                            p1 = contour_points[i]
                            p2 = contour_points[(i + 1) % len(contour_points)]
                            
                            # Calculate distance between consecutive contour points
                            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            actual_contour_length += distance
                        
                        # Convert to cm
                        actual_contour_length_cm = actual_contour_length * pixel_cm_ratio
                        
                        # Use actual contour length for total
                        total_distance = actual_contour_length_cm
                        distances = corner_distances  # Keep corner distances for display
                        
                        # Draw distance text with better positioning
                        for j in range(len(corner_points)):
                            p1 = corner_points[j]
                            p2 = corner_points[(j + 1) % len(corner_points)]
                            
                            # Calculate distance for display
                            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                            pixel_cm_ratio = self.settings['pixel_cm_ratio']
                            distance_cm = distance * pixel_cm_ratio
                            
                            mid_x = (p1[0] + p2[0]) // 2
                            mid_y = (p1[1] + p2[1]) // 2
                            
                            # Offset text position to avoid overlapping with line
                            offset_x = 0
                            offset_y = 0
                            if abs(p2[0] - p1[0]) > abs(p2[1] - p1[1]):  # Horizontal line
                                offset_y = -30
                            else:  # Vertical line
                                offset_x = 30
                            
                            distance_text = f'{distance_cm:.1f}cm'
                            (text_width, text_height), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)
                            
                            # Background for distance text
                            cv2.rectangle(result_frame, 
                                        (mid_x + offset_x - text_width//2 - 8, mid_y + offset_y - text_height//2 - 8), 
                                        (mid_x + offset_x + text_width//2 + 8, mid_y + offset_y + text_height//2 + 8), 
                                        (0, 0, 0), -1)
                            
                            # Distance text
                            cv2.putText(result_frame, distance_text, 
                                      (mid_x + offset_x - text_width//2, mid_y + offset_y + text_height//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)  # Yellow text
                        
                        # Summary information removed - only shown in side panel
                        
                        measurements = {
                            'corners': len(corner_points),
                            'segments': len(distances),
                            'total_perimeter': total_distance,
                            'distances': distances
                        }
                        break
        
        return result_frame, measurements

class ModernObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Modern Object Detection & Measurement")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize models
        self.yolo_model = None
        self.birefnet_model = None
        self.birefnet_transform = None
        self.load_models()
        
        # Video variables
        self.video_thread = None
        self.processing_thread = None
        self.current_frame = None
        self.last_processed_frame = None  # Store last processed frame
        self.last_measurements = None     # Store last measurements
        
        # Settings
        self.settings = {
            'confidence': 0.8,
            'contour_threshold': 100,
            'epsilon': 0.015,
            'pixel_cm_ratio': 0.061  # DJI Osmo Pocket 3 @ 87.6cm height (calibrated)
        }
        
        self.setup_ui()
        self.setup_styles()
        
    def load_models(self):
        """Load YOLOv11 and BirefNet models"""
        try:
            # Set torch precision
            torch.set_float32_matmul_precision("medium")
            
            # Load YOLOv11 model
            yolo_model_path = "models/yolov11-small-cloths.pt"
            if os.path.exists(yolo_model_path):
                self.yolo_model = YOLO(yolo_model_path)
                print("YOLOv11 model loaded successfully")
            else:
                print(f"YOLOv11 model not found at {yolo_model_path}")
                
            # Load BirefNet model
            try:
                self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet_dynamic", trust_remote_code=True
                )
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.birefnet_model.to(device)
                self.birefnet_model.eval()
                
                self.birefnet_transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                print(f"BirefNet model loaded successfully on {device}")
                
            except Exception as e:
                print(f"Error loading BirefNet model: {e}")
                # Set to None so we can handle it gracefully
                self.birefnet_model = None
                self.birefnet_transform = None
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #666; background-color: #000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No Video")
        
        left_panel.addWidget(self.video_label)
        
        # Video controls
        video_controls = QHBoxLayout()
        
        self.webcam_btn = QPushButton("Webcam")
        self.webcam_btn.clicked.connect(self.open_webcam)
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        
        video_controls.addWidget(self.webcam_btn)
        video_controls.addWidget(self.load_video_btn)
        video_controls.addWidget(self.stop_btn)
        video_controls.addStretch()
        
        left_panel.addLayout(video_controls)
        
        # Right panel - Controls and results
        right_panel = QVBoxLayout()
        
        # Calculate button
        self.calculate_btn = QPushButton("Calculate Measurements")
        self.calculate_btn.setMinimumHeight(50)
        self.calculate_btn.clicked.connect(self.calculate_measurements)
        right_panel.addWidget(self.calculate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Confidence slider
        conf_label = QLabel("Confidence:")
        conf_label.setToolTip("Object detection confidence threshold\nLow value: Detects more objects (may increase false positives)\nHigh value: Only detects certain objects (some objects may be missed)")
        settings_layout.addWidget(conf_label, 0, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(80)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_slider.setToolTip("Object detection confidence threshold (0.10-1.00)")
        settings_layout.addWidget(self.confidence_slider, 0, 1)
        self.confidence_value_label = QLabel("0.80")
        self.confidence_value_label.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 40px;")
        settings_layout.addWidget(self.confidence_value_label, 0, 2)
        
        # Contour threshold slider
        contour_label = QLabel("Contour Threshold:")
        contour_label.setToolTip("Minimum area threshold for contours\nLow value: Detects small objects too (may increase noise)\nHigh value: Only detects large objects (small details may be lost)")
        settings_layout.addWidget(contour_label, 1, 0)
        self.contour_slider = QSlider(Qt.Horizontal)
        self.contour_slider.setRange(10, 1000)
        self.contour_slider.setValue(100)
        self.contour_slider.valueChanged.connect(self.update_contour_threshold)
        self.contour_slider.setToolTip("Minimum area threshold for contours (10-1000 pixels)")
        settings_layout.addWidget(self.contour_slider, 1, 1)
        self.contour_value_label = QLabel("100")
        self.contour_value_label.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 40px;")
        settings_layout.addWidget(self.contour_value_label, 1, 2)
        
        # Epsilon slider
        epsilon_label = QLabel("Corner Detection:")
        epsilon_label.setToolTip("Corner detection sensitivity\nLow value: More sensitive, detects many corners (detailed shapes)\nHigh value: Less sensitive, detects fewer corners (general shape)")
        settings_layout.addWidget(epsilon_label, 2, 0)
        self.epsilon_slider = QSlider(Qt.Horizontal)
        self.epsilon_slider.setRange(5, 50)
        self.epsilon_slider.setValue(15)
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        self.epsilon_slider.setToolTip("Corner detection sensitivity (0.005-0.050)")
        settings_layout.addWidget(self.epsilon_slider, 2, 1)
        self.epsilon_value_label = QLabel("0.015")
        self.epsilon_value_label.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 40px;")
        settings_layout.addWidget(self.epsilon_value_label, 2, 2)
        
        # Pixel to cm ratio slider
        ratio_label = QLabel("Pixel/CM Ratio:")
        ratio_label.setToolTip("Pixel to centimeter conversion ratio\nLow value: Larger measurements (1 pixel = more cm)\nHigh value: Smaller measurements (1 pixel = less cm)\nDJI Osmo Pocket 3 @ 87.6cm: 0.061 = 1 pixel = 0.061 cm")
        settings_layout.addWidget(ratio_label, 3, 0)
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(5, 100)  # Lower minimum for DJI Osmo Pocket 3
        self.ratio_slider.setValue(6)  # 0.061 * 100 ≈ 6
        self.ratio_slider.valueChanged.connect(self.update_pixel_cm_ratio)
        self.ratio_slider.setToolTip("Pixel to centimeter conversion ratio (0.05-1.00)\nDJI Osmo Pocket 3 @ 87.6cm height: ~6 (0.061)")
        settings_layout.addWidget(self.ratio_slider, 3, 1)
        self.ratio_value_label = QLabel("0.061")
        self.ratio_value_label.setStyleSheet("color: #ffd700; font-weight: bold; min-width: 40px;")
        settings_layout.addWidget(self.ratio_value_label, 3, 2)
        
        right_panel.addWidget(settings_group)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_label = QLabel("""
        <div style='background-color: #2d2d2d; padding: 15px; border-radius: 8px;'>
            <h3 style='color: #4a90e2; margin: 0 0 10px 0;'>Ready to Measure</h3>
            <p style='color: #cccccc; margin: 0;'>
                1. Open webcam or load a video<br>
                2. Adjust settings if needed<br>
                3. Click "Calculate Measurements"<br>
                4. View results here and on the image
            </p>
        </div>
        """)
        self.results_label.setWordWrap(True)
        self.results_label.setTextFormat(Qt.RichText)
        results_layout.addWidget(self.results_label)
        
        right_panel.addWidget(results_group)
        
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def setup_styles(self):
        """Setup modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #4a90e2;
                border: none;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2d5aa0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            
            QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #666666;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #5c6bc0;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 2px solid #666;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-top: 1px solid #666;
            }
        """)
    
    def open_webcam(self):
        """Open webcam"""
        self.stop_video()
        self.video_thread = VideoThread(0)
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.video_thread.start()
        self.statusBar().showMessage("Webcam opened")
    
    def load_video(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
        )
        
        if file_path:
            self.stop_video()
            # For video files, we'll use a different approach
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.timer = QTimer()
                self.timer.timeout.connect(self.read_video_frame)
                self.timer.start(33)  # ~30 FPS
                self.statusBar().showMessage(f"Video loaded: {os.path.basename(file_path)}")
            else:
                self.statusBar().showMessage("Could not open video file")
    
    def stop_video(self):
        """Stop video playback"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        if hasattr(self, 'cap'):
            self.cap.release()
            self.cap = None
        
        self.current_frame = None
        self.last_processed_frame = None  # Clear processed frame
        self.last_measurements = None     # Clear measurements
        self.video_label.setText("No Video")
        self.statusBar().showMessage("Video stopped")
    
    def update_video_frame(self, frame):
        """Update video frame from thread"""
        self.current_frame = frame
        
        # If we have a processed frame, show it instead of raw frame
        if self.last_processed_frame is not None:
            self.display_frame(self.last_processed_frame)
        else:
            self.display_frame(frame)
    
    def read_video_frame(self):
        """Read frame from video file"""
        if hasattr(self, 'cap') and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                
                # If we have a processed frame, show it instead of raw frame
                if self.last_processed_frame is not None:
                    self.display_frame(self.last_processed_frame)
                else:
                    self.display_frame(frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame(self, frame):
        """Display frame on label"""
        if frame is None:
            return
        
        # Convert frame to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def calculate_measurements(self):
        """Calculate object measurements"""
        if self.current_frame is None:
            self.statusBar().showMessage("No frame available for calculation")
            return
        
        if self.yolo_model is None:
            self.statusBar().showMessage("YOLO model not loaded")
            return
        
        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.current_frame, 
            self.yolo_model, 
            self.birefnet_model, 
            self.birefnet_transform, 
            self.settings
        )
        
        self.processing_thread.result_ready.connect(self.on_processing_complete)
        self.processing_thread.progress_updated.connect(self.update_progress)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calculate_btn.setEnabled(False)
        
        self.processing_thread.start()
        self.statusBar().showMessage("Processing...")
    
    def on_processing_complete(self, result_frame, measurements):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        
        if 'error' in measurements:
            self.results_label.setText(f"Error: {measurements['error']}")
            self.statusBar().showMessage("Processing failed")
        else:
            # Store the processed frame and measurements
            self.last_processed_frame = result_frame.copy()
            self.last_measurements = measurements
            
            # Display result frame
            self.display_frame(result_frame)
            
            # Update results with more detailed information
            result_text = f"""
            <div style='background-color: #2d2d2d; padding: 15px; border-radius: 8px;'>
                <h3 style='color: #4a90e2; margin: 0 0 10px 0;'>Measurement Results</h3>
                
                <div style='background-color: #3d3d3d; padding: 10px; border-radius: 5px; margin: 10px 0; line-height: 1.6;'>
                    <b style='color: #ffffff;'>Summary:</b><br>
                    • <span style='color: #4a90e2;'>Corners:</span> {measurements['corners']}<br>
                    • <span style='color: #4a90e2;'>Segments:</span> {measurements['segments']}<br>
                    • <span style='color: #4a90e2;'>Actual Perimeter:</span> <b style='color: #ffd700;'>{measurements['total_perimeter']:.1f} cm</b><br>
                    • <span style='color: #4a90e2;'>Scale:</span> {self.settings['pixel_cm_ratio']:.2f} px/cm
                </div>
                
                <div style='background-color: #3d3d3d; padding: 10px; border-radius: 5px; line-height: 1.6;'>
                    <b style='color: #ffffff;'>Corner Segment Details:</b><br>
            """
            
            for i, distance in enumerate(measurements['distances']):
                result_text += f"• <span style='color: #4a90e2;'>Segment {i+1}:</span> <b style='color: #ffd700;'>{distance:.1f} cm</b><br>"
            
            result_text += "</div></div>"
            
            self.results_label.setText(result_text)
            self.statusBar().showMessage("Measurements calculated")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_confidence(self, value):
        """Update confidence setting"""
        self.settings['confidence'] = value / 100.0
        self.confidence_value_label.setText(f"{value / 100.0:.2f}")
    
    def update_contour_threshold(self, value):
        """Update contour threshold setting"""
        self.settings['contour_threshold'] = value
        self.contour_value_label.setText(f"{value}")
    
    def update_epsilon(self, value):
        """Update epsilon setting"""
        self.settings['epsilon'] = value / 1000.0
        self.epsilon_value_label.setText(f"{value / 1000.0:.3f}")
    
    def update_pixel_cm_ratio(self, value):
        """Update pixel to cm ratio setting"""
        self.settings['pixel_cm_ratio'] = value / 100.0
        self.ratio_value_label.setText(f"{value / 100.0:.2f}")
    

    def closeEvent(self, event):
        """Handle application close"""
        self.stop_video()
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ModernObjectDetectionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()