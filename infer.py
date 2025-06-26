import cv2
import os
from pathlib import Path
import time
import traceback
import numpy as np
from collections import deque
from ultralytics import YOLO
from rembg import remove
from PIL import Image

# Load the YOLO model
model = YOLO("models/yolov11_small.onnx")

def smooth_contours(contours, epsilon_factor=0.01):
    """Smooth contours using approxPolyDP"""
    smoothed = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        smoothed.append(approx)
    return smoothed

def process_video(video_path="videos\DJI_20250620150908_0208_D.MP4"):
    cap = cv2.VideoCapture(video_path)
    
    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions (half of original size)
    new_width = width // 2
    new_height = height // 2
    
    # Variables for prediction tracking
    last_bbox = None
    frame_count = 0
    prediction_interval = 10  # Run prediction every N frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Only run prediction every N frames
        if frame_count % prediction_interval == 0:
            results = model.predict(source=frame, conf=0.70, iou=0.45, verbose=False)
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the first detection
                box = result.boxes[0]
                last_bbox = list(map(int, box.xyxy[0].cpu().numpy()))
        
        frame_count += 1
        
        # Process frame if we have a detection
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(roi_rgb)
            
            # Remove background
            output_pil = remove(pil_image)
            
            # Convert back to numpy array
            output_array = np.array(output_pil)
            
            # Convert RGBA to BGR for OpenCV
            output_rgb = output_array[:, :, :3]
            output_mask = output_array[:, :, 3]
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            
            # Create output frame
            output = frame.copy()
            
            # Create a full frame mask
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = output_mask
            
            # Apply the mask to dim the background
            output[full_mask == 0] = output[full_mask == 0] * 0.3
            
            # Place the segmented object
            output[y1:y2, x1:x2][output_mask > 0] = output_bgr[output_mask > 0]
            
            # Draw the YOLO bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Show results
            cv2.imshow('Segmentation Mask', full_mask)
            cv2.imshow('Result', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Starting frame-by-frame video inference with YOLO11...")
    process_video()

