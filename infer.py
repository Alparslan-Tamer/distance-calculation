import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import os
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class ObjectSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Segmentation")
        self.root.geometry("1400x800")
        
        # Model paths
        self.yolo_model_path = "models/yolov11-small-cloths.pt"
        
        # Initialize models
        self.yolo_model = None
        self.birefnet_model = None
        self.birefnet_transform = None
        self.load_models()
        
        # Video variables
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.processed_frame = None
        
        # Video recording variables
        self.is_recording = False
        self.video_writer = None
        self.output_video_path = None
        
        # Video processing variables
        self.detection_results = None
        self.segmentation_mask = None
        
        # Detection variables
        self.detection_results = None
        self.segmentation_mask = None
        
        # Continuous processing
        self.is_continuous = False
        
        self.setup_ui()
        
    def load_models(self):
        """Load YOLOv11 and BirefNet models"""
        try:
            # Set torch precision for better performance
            torch.set_float32_matmul_precision("medium")
            
            # Load YOLOv11 model
            if os.path.exists(self.yolo_model_path):
                self.yolo_model = YOLO(self.yolo_model_path)
                print("YOLOv11 model loaded successfully")
            else:
                print(f"YOLOv11 model not found at {self.yolo_model_path}")
                
            # Load BirefNet model from HuggingFace
            try:
                self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet_dynamic", trust_remote_code=True
                )
                
                # Move to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.birefnet_model.to(device)
                self.birefnet_model.eval()
                
                # Create transform
                self.birefnet_transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                print(f"BirefNet model loaded successfully on {device}")
                
            except Exception as e:
                print(f"Error loading BirefNet model: {e}")
                messagebox.showerror("Error", f"Failed to load BirefNet model: {e}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            messagebox.showerror("Error", f"Failed to load models: {e}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Object Detection & Segmentation", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Left panel - Input video
        left_frame = ttk.LabelFrame(main_frame, text="Input Video", padding="10")
        left_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.input_canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.input_canvas.pack()
        
        # Input controls
        input_controls = ttk.Frame(left_frame)
        input_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(input_controls, text="Open Webcam", command=self.open_webcam).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_controls, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_controls, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=(0, 5))
        
        # Recording controls
        ttk.Button(input_controls, text="Start Recording", command=self.start_recording).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(input_controls, text="Stop Recording", command=self.stop_recording).pack(side=tk.LEFT)
        
        # Right panel - Output video
        right_frame = ttk.LabelFrame(main_frame, text="Original with Measurements", padding="10")
        right_frame.grid(row=1, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Original with measurements panel
        self.original_canvas = tk.Canvas(right_frame, width=640, height=480, bg="black")
        self.original_canvas.pack()
        
        # Output controls
        output_controls = ttk.Frame(right_frame)
        output_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(output_controls, text="Calculate", command=self.calculate_segmentation).pack(side=tk.LEFT, padx=(0, 5))
        
        # Continuous processing
        self.continuous_var = tk.BooleanVar()
        ttk.Checkbutton(output_controls, text="Continuous", variable=self.continuous_var, 
                       command=self.toggle_continuous).pack(side=tk.LEFT, padx=(0, 5))
        
        # Confidence slider
        ttk.Label(output_controls, text="Conf:").pack(side=tk.LEFT, padx=(10, 0))
        self.confidence_var = tk.DoubleVar(value=0.8)
        self.confidence_scale = ttk.Scale(output_controls, from_=0.1, to=1.0, 
                                        variable=self.confidence_var, orient=tk.HORIZONTAL, length=100)
        self.confidence_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # Contour threshold slider
        ttk.Label(output_controls, text="Contour:").pack(side=tk.LEFT, padx=(10, 0))
        self.contour_threshold_var = tk.IntVar(value=100)
        self.contour_scale = ttk.Scale(output_controls, from_=10, to=1000, 
                                     variable=self.contour_threshold_var, orient=tk.HORIZONTAL, length=100)
        self.contour_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # Corner detection settings
        ttk.Label(output_controls, text="Epsilon:").pack(side=tk.LEFT, padx=(10, 0))
        self.epsilon_var = tk.DoubleVar(value=0.02)
        self.epsilon_scale = ttk.Scale(output_controls, from_=0.01, to=0.1, 
                                     variable=self.epsilon_var, orient=tk.HORIZONTAL, length=80)
        self.epsilon_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # Pixel to cm ratio
        ttk.Label(output_controls, text="Px/cm:").pack(side=tk.LEFT, padx=(10, 0))
        self.pixel_cm_ratio_var = tk.DoubleVar(value=0.25)
        self.pixel_cm_scale = ttk.Scale(output_controls, from_=0.1, to=1.0, 
                                      variable=self.pixel_cm_ratio_var, orient=tk.HORIZONTAL, length=80)
        self.pixel_cm_scale.pack(side=tk.LEFT, padx=(5, 0))
        
        # Performance panel
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Metrics", padding="5")
        perf_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 5))
        
        # Performance labels
        self.detection_time_var = tk.StringVar(value="Detection: --")
        self.segmentation_time_var = tk.StringVar(value="Segmentation: --")
        self.total_time_var = tk.StringVar(value="Total: --")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.contour_info_var = tk.StringVar(value="Contours: --")
        self.corner_info_var = tk.StringVar(value="Corners: --")
        
        ttk.Label(perf_frame, textvariable=self.detection_time_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(perf_frame, textvariable=self.segmentation_time_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(perf_frame, textvariable=self.total_time_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(perf_frame, textvariable=self.fps_var, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(perf_frame, textvariable=self.contour_info_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(perf_frame, textvariable=self.corner_info_var, font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Recording indicator
        self.recording_indicator = ttk.Label(status_frame, text="⏺ REC", foreground="red", font=("Arial", 9, "bold"))
        self.recording_indicator.pack(side=tk.RIGHT, padx=(5, 0))
        self.recording_indicator.pack_forget()  # Hide initially
        
    def open_webcam(self):
        """Open webcam for live video"""
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_playing = True
            self.status_var.set("Webcam opened")
            self.play_video()
        else:
            messagebox.showerror("Error", "Could not open webcam")
            
    def load_video(self):
        """Load video file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv, *.MP4"), ("All files", "*.*")]
            )
            
            if file_path:
                # Force stop any current video and clean up
                self.force_stop_video()
                
                # Clear all references
                self.current_frame = None
                self.processed_frame = None
                self.detection_results = None
                self.segmentation_mask = None
                
                # Longer delay to ensure complete cleanup
                time.sleep(0.5)
                
                # Load new video
                self.cap = cv2.VideoCapture(file_path)
                if self.cap.isOpened():
                    self.is_playing = True
                    self.status_var.set(f"Video loaded: {os.path.basename(file_path)}")
                    self.play_video()
                else:
                    messagebox.showerror("Error", "Could not open video file")
                    self.cap = None
        except Exception as e:
            print(f"Error loading video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {e}")
            self.cap = None
                
    def stop_video(self):
        """Stop video playback"""
        try:
            self.is_playing = False
            
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Release video capture
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Clear frame references
            self.current_frame = None
            self.processed_frame = None
            
            # Clear canvas
            if hasattr(self, 'input_canvas'):
                self.input_canvas.delete("all")
            if hasattr(self, 'original_canvas'):
                self.original_canvas.delete("all")
                
            self.status_var.set("Video stopped")
            
        except Exception as e:
            print(f"Error stopping video: {e}")
            self.status_var.set("Error stopping video")
    
    def force_stop_video(self):
        """Force stop video with aggressive cleanup"""
        try:
            # Stop all processing
            self.is_playing = False
            self.is_continuous = False
            
            # Stop recording
            if self.is_recording:
                self.stop_recording()
            
            # Release video capture with multiple attempts
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            
            # Release video writer
            if hasattr(self, 'video_writer') and self.video_writer:
                try:
                    self.video_writer.release()
                except:
                    pass
                self.video_writer = None
            
            # Clear all references
            self.current_frame = None
            self.processed_frame = None
            self.detection_results = None
            self.segmentation_mask = None
            
            # Clear canvas with error handling
            try:
                if hasattr(self, 'input_canvas'):
                    self.input_canvas.delete("all")
                if hasattr(self, 'original_canvas'):
                    self.original_canvas.delete("all")
            except:
                pass
                
            # Force garbage collection
            import gc
            gc.collect()
            
            self.status_var.set("Video force stopped")
            
        except Exception as e:
            print(f"Error in force stop video: {e}")
            self.status_var.set("Error in force stop")
        
    def start_recording(self):
        """Start recording output video"""
        if self.is_recording:
            messagebox.showinfo("Info", "Already recording")
            return
            
        # Ask for output file path
        file_path = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if file_path:
            self.output_video_path = file_path
            self.is_recording = True
            self.status_var.set(f"Recording started: {os.path.basename(file_path)}")
            self.recording_indicator.pack(side=tk.RIGHT, padx=(5, 0))  # Show recording indicator
        else:
            self.status_var.set("Recording cancelled")
            
    def initialize_video_writer(self, frame_shape):
        """Initialize video writer with given frame shape"""
        if not self.is_recording or not self.output_video_path:
            return
            
        # Get frame dimensions
        height, width = frame_shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, 
            fourcc, 
            30.0,  # FPS
            (width, height)
        )
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Could not initialize video writer")
            self.is_recording = False
            self.output_video_path = None
            
    def stop_recording(self):
        """Stop recording output video"""
        if not self.is_recording:
            messagebox.showinfo("Info", "Not recording")
            return
            
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        if self.output_video_path:
            self.status_var.set(f"Recording saved: {os.path.basename(self.output_video_path)}")
            self.output_video_path = None
        else:
            self.status_var.set("Recording stopped")
            
        # Hide recording indicator
        self.recording_indicator.pack_forget()
        
    def play_video(self):
        """Play video in a separate thread"""
        def video_loop():
            try:
                while self.is_playing and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        # Video ended, restart if webcam
                        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0:
                            break
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                        
                    self.current_frame = frame.copy()  # Make a copy to avoid reference issues
                    self.display_frame(frame, self.input_canvas)
                    time.sleep(1/30)  # 30 FPS
            except Exception as e:
                print(f"Error in video loop: {e}")
                self.is_playing = False
                
        threading.Thread(target=video_loop, daemon=True).start()
        
    def display_frame(self, frame, canvas):
        """Display frame on canvas"""
        try:
            if frame is None or canvas is None:
                return
                
            # Resize frame to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas.image = img_tk  # Keep a reference
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
    def toggle_continuous(self):
        """Toggle continuous processing mode"""
        self.is_continuous = self.continuous_var.get()
        if self.is_continuous:
            self.status_var.set("Continuous processing enabled")
            self.start_continuous_processing()
        else:
            self.status_var.set("Continuous processing disabled")
            
    def start_continuous_processing(self):
        """Start continuous processing"""
        def continuous_loop():
            while self.is_continuous and self.current_frame is not None:
                self._process_frame()
                
                # Save frame if recording (for continuous mode)
                if self.is_recording and self.processed_frame is not None:
                    # Initialize video writer if not already done
                    if self.video_writer is None:
                        self.initialize_video_writer(self.processed_frame.shape)
                    
                    # Write frame to video
                    if self.video_writer and self.video_writer.isOpened():
                        self.video_writer.write(self.processed_frame)
                
                time.sleep(0.1)  # 10 FPS for continuous processing
                
        threading.Thread(target=continuous_loop, daemon=True).start()
        
    def calculate_segmentation(self):
        """Calculate object detection and segmentation"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return
            
        self.status_var.set("Processing...")
        
        # Run processing in separate thread
        threading.Thread(target=self._process_frame, daemon=True).start()
        
    def _process_frame(self):
        """Process frame with YOLOv11 and BirefNet"""
        try:
            import time
            frame = self.current_frame.copy()
            total_start_time = time.time()
            
            # Step 1: YOLOv11 Detection
            if self.yolo_model:
                detection_start_time = time.time()
                confidence = self.confidence_var.get()
                results = self.yolo_model(frame, conf=confidence)
                detection_time = time.time() - detection_start_time
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get the first detection
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Crop the detected object
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        # Step 2: BirefNet Segmentation
                        segmentation_start_time = time.time()
                        mask = self.run_birefnet_inference(cropped)
                        segmentation_time = time.time() - segmentation_start_time
                        
                        # Step 3: Apply mask to original frame and draw contours
                        postprocess_start_time = time.time()
                        # Resize mask to match cropped size
                        mask_resized = cv2.resize(mask, (cropped.shape[1], cropped.shape[0]))
                        
                        # Create full frame mask
                        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        full_mask[y1:y2, x1:x2] = mask_resized
                        
                        # Initialize original frame with measurements early
                        original_frame_with_measurements = frame.copy()
                        
                        # Clean mask for better contours
                        # Apply morphological operations to clean noise
                        kernel = np.ones((5,5), np.uint8)
                        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
                        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
                        
                        # Find contours from cleaned mask
                        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Create output frame with contours
                        segmented_frame = frame.copy()
                        
                        # Draw contours on the frame
                        if contours:
                            # Sort contours by area (largest first)
                            contours = sorted(contours, key=cv2.contourArea, reverse=True)
                            
                            # Draw all significant contours
                            for i, contour in enumerate(contours):
                                area = cv2.contourArea(contour)
                                
                                # Only draw contours with significant area
                                contour_threshold = self.contour_threshold_var.get()
                                if area > contour_threshold:  # Minimum area threshold
                                    # Different colors for different contours
                                    if i == 0:  # Largest contour
                                        color = (0, 255, 0)  # Green
                                        thickness = 3
                                    else:
                                        color = (0, 255, 255)  # Yellow
                                        thickness = 2
                                    
                                                                            # Draw contour with thicker lines (only on original frame)
                                        cv2.drawContours(original_frame_with_measurements, [contour], -1, color, thickness + 1)
                                    
                                    # Detect and draw corner points for largest contour
                                    if i == 0:
                                        # Contour approximation for corner detection
                                        epsilon_value = self.epsilon_var.get()
                                        epsilon = epsilon_value * cv2.arcLength(contour, True)
                                        approx = cv2.approxPolyDP(contour, epsilon, True)
                                        
                                        # Ensure we have at least 3 points for a meaningful shape
                                        if len(approx) < 3:
                                            print(f"Warning: Too few approximation points ({len(approx)}), adjusting epsilon")
                                            # Try with smaller epsilon
                                            epsilon = epsilon_value * 0.5 * cv2.arcLength(contour, True)
                                            approx = cv2.approxPolyDP(contour, epsilon, True)
                                            if len(approx) < 3:
                                                print(f"Still too few points ({len(approx)}), using original contour")
                                                approx = contour
                                        
                                        # Additional safety check
                                        if len(approx) < 2:
                                            print(f"Error: Contour has only {len(approx)} points, cannot calculate distances")
                                            self.root.after(0, lambda: self.corner_info_var.set(f"Error: Need at least 2 points"))
                                            break
                                        
                                        # Draw corner points and calculate distances
                                        corner_points = []
                                        for i, point in enumerate(approx):
                                            x, y = point.ravel()
                                            corner_points.append((int(x), int(y)))
                                        
                                        # Draw corner points with numbers
                                        for i, point in enumerate(corner_points):
                                            # Draw larger corner points with numbers
                                            cv2.circle(original_frame_with_measurements, point, 12, (0, 0, 255), -1)  # Red circles
                                            
                                            # Add corner number
                                            cv2.putText(original_frame_with_measurements, f'C{i+1}', (point[0] + 15, point[1] + 8), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                                        
                                        # Calculate distances between consecutive points
                                        distances = []
                                        total_distance = 0
                                        
                                        # Check if we have enough points for distance calculation
                                        if len(corner_points) < 2:
                                            print(f"Not enough corner points: {len(corner_points)}")
                                            self.root.after(0, lambda: self.corner_info_var.set(f"Corners: {len(approx)} | Need at least 2 points"))
                                            break
                                        
                                        for i in range(len(corner_points)):
                                            # Get current and next point (circular)
                                            p1 = corner_points[i]
                                            p2 = corner_points[(i + 1) % len(corner_points)]
                                            
                                            # Calculate simple Euclidean distance between corner points
                                            final_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                                            distances.append(final_distance)
                                            total_distance += final_distance
                                            
                                            # Add distance text directly at midpoint
                                            mid_x = (p1[0] + p2[0]) // 2
                                            mid_y = (p1[1] + p2[1]) // 2
                                            pixel_cm_ratio = self.pixel_cm_ratio_var.get()
                                            distance_cm = final_distance * pixel_cm_ratio  # Convert to cm
                                            
                                            # Create simple distance text
                                            distance_text = f'{distance_cm:.1f}cm'
                                            
                                            # Add distance text with large font
                                            (text_width, text_height), baseline = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                                            
                                            # Draw background for better visibility
                                            cv2.rectangle(original_frame_with_measurements, 
                                                        (mid_x - text_width//2 - 5, mid_y - text_height//2 - 5), 
                                                        (mid_x + text_width//2 + 5, mid_y + text_height//2 + 5), 
                                                        (0, 0, 0), -1)
                                            
                                            # Add distance text with large font
                                            cv2.putText(original_frame_with_measurements, distance_text, 
                                                      (mid_x - text_width//2, mid_y + text_height//2), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                                            
                                            print(f"Segment {i+1}: {distance_cm:.1f}cm")
                                        
                                        pixel_cm_ratio = self.pixel_cm_ratio_var.get()
                                        print(f"Contour corners: {len(approx)}")
                                        print(f"Total perimeter: {total_distance * pixel_cm_ratio:.1f}cm")
                                        print(f"Individual distances: {[f'{d * pixel_cm_ratio:.1f}cm' for d in distances]}")
                                        print(f"Contour has {len(contour)} points, total contour length: {cv2.arcLength(contour, True) * pixel_cm_ratio:.1f}cm")
                                        print(f"Measurements shown on both segmented and original frames")
                                        print(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")
                                        
                                        # Update corner info in GUI
                                        self.root.after(0, lambda: self.corner_info_var.set(f"Corners: {len(approx)} | Perimeter: {total_distance * pixel_cm_ratio:.1f}cm"))
                                        
                                        # Draw bounding box (removed - keeping only corner points and segments)
                                        
                                        # Add total perimeter information with details
                                        total_perimeter_cm = total_distance * pixel_cm_ratio
                                        perimeter_text = f'Total: {total_perimeter_cm:.1f}cm'
                                        
                                        # Add perimeter text directly
                                        cv2.putText(original_frame_with_measurements, perimeter_text, (10, 40), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                                        
                                        # Add segment details in bottom right (simplified)
                                        if distances:
                                            # Calculate position for bottom right
                                            frame_height = original_frame_with_measurements.shape[0]
                                            detail_y_start = frame_height - 30
                                            
                                            # Add simple segment count
                                            segment_text = f'Segments: {len(distances)}'
                                            cv2.putText(original_frame_with_measurements, segment_text, 
                                                      (original_frame_with_measurements.shape[1] - 180, detail_y_start), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                            
                            contour_count = len([c for c in contours if cv2.contourArea(c) > contour_threshold])
                            largest_area = cv2.contourArea(contours[0])
                            print(f"Contours found: {contour_count} contours, largest area: {largest_area:.0f}")
                            
                            # Update contour info in GUI
                            self.root.after(0, lambda: self.contour_info_var.set(f"Contours: {contour_count} (Area: {largest_area:.0f})"))
                        else:
                            print("No contours found in mask")
                            self.root.after(0, lambda: self.contour_info_var.set("Contours: 0"))
                            self.root.after(0, lambda: self.corner_info_var.set("Corners: 0"))
                        postprocess_time = time.time() - postprocess_start_time
                        
                        total_time = time.time() - total_start_time
                        
                        # Update performance metrics
                        self.root.after(0, lambda: self.detection_time_var.set(f"Detection: {detection_time:.3f}s"))
                        self.root.after(0, lambda: self.segmentation_time_var.set(f"Segmentation: {segmentation_time:.3f}s"))
                        self.root.after(0, lambda: self.total_time_var.set(f"Total: {total_time:.3f}s"))
                        self.root.after(0, lambda: self.fps_var.set(f"FPS: {1/total_time:.1f}"))
                        
                        self.processed_frame = original_frame_with_measurements
                        self.root.after(0, lambda: self.display_frame(original_frame_with_measurements, self.original_canvas))
                        self.root.after(0, lambda: self.status_var.set("Processing completed successfully"))
                        
                        # Save frame if recording
                        if self.is_recording and self.processed_frame is not None:
                            # Initialize video writer if not already done
                            if self.video_writer is None:
                                self.initialize_video_writer(self.processed_frame.shape)
                            
                            # Write frame to video
                            if self.video_writer and self.video_writer.isOpened():
                                self.video_writer.write(self.processed_frame)
                        
                        # Print detailed timing to console
                        print(f"\n=== PERFORMANCE METRICS ===")
                        print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
                        print(f"Crop size: {cropped.shape[1]}x{cropped.shape[0]}")
                        print(f"YOLOv11 Detection: {detection_time:.3f}s ({detection_time*1000:.1f}ms)")
                        print(f"BirefNet Segmentation: {segmentation_time:.3f}s ({segmentation_time*1000:.1f}ms)")
                        print(f"Post-processing: {postprocess_time:.3f}s ({postprocess_time*1000:.1f}ms)")
                        print(f"Total processing time: {total_time:.3f}s ({total_time*1000:.1f}ms)")
                        print(f"FPS: {1/total_time:.1f}")
                        print("=" * 30)
                        
                    else:
                        # Create empty original frame with measurements
                        original_frame_with_measurements = frame.copy()
                        self.root.after(0, lambda: self.display_frame(original_frame_with_measurements, self.original_canvas))
                        self.root.after(0, lambda: messagebox.showinfo("Info", "No valid object detected"))
                else:
                    detection_time = time.time() - detection_start_time
                    total_time = time.time() - total_start_time
                    
                    # Create empty original frame with measurements
                    original_frame_with_measurements = frame.copy()
                    self.root.after(0, lambda: self.display_frame(original_frame_with_measurements, self.original_canvas))
                    
                    # Update performance metrics for detection only
                    self.root.after(0, lambda: self.detection_time_var.set(f"Detection: {detection_time:.3f}s"))
                    self.root.after(0, lambda: self.segmentation_time_var.set("Segmentation: --"))
                    self.root.after(0, lambda: self.total_time_var.set(f"Total: {total_time:.3f}s"))
                    self.root.after(0, lambda: self.fps_var.set(f"FPS: {1/total_time:.1f}"))
                    
                    self.root.after(0, lambda: messagebox.showinfo("Info", f"No objects detected (Detection time: {detection_time:.3f}s)"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "YOLOv11 model not loaded"))
                
        except Exception as e:
            print(f"Error in processing: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            self.root.after(0, lambda: self.status_var.set("Processing failed"))
            
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
        
    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def __del__(self):
        """Destructor to clean up resources"""
        try:
            self.force_stop_video()
        except:
            pass

def main():
    root = tk.Tk()
    app = ObjectSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
