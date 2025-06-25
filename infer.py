import cv2
import os
from pathlib import Path
import time
import traceback
import numpy as np
from collections import deque

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/alp/distance-calculation/models/yolov11_small.onnx")

def process_videos():
    """Process all videos in the videos folder frame by frame"""
    
    # Define paths
    videos_dir = Path("videos")
    output_dir = Path("output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
    
    if not video_files:
        print("No video files found in the videos folder!")
        return
    
    print(f"Found {len(video_files)} video files to process:")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"❌ Error: Could not open video {video_path.name}")
                continue
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"   Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Create output video writer
            output_video_path = output_dir / f"processed_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            # Create output text file for detections
            output_txt_path = output_dir / f"detections_{video_path.stem}.txt"
            
            frame_count = 0
            processed_frames = 0
            start_time = time.time()
            
            # Store previous detections (frame_number, detections_list)
            previous_detections = []
            frames_without_detection = 0
            max_frames_without_detection = 30  # Show previous detections for 30 frames
            
            with open(output_txt_path, 'w') as txt_file:
                txt_file.write("frame_number,class_id,confidence,x1,y1,x2,y2\n")
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process every 10th frame (to speed up processing)
                    if frame_count % 10 != 0:
                        # Use previous detections for this frame
                        annotated_frame = frame.copy()
                        
                        if previous_detections:
                            for detection in previous_detections:
                                x1, y1, x2, y2, class_id, confidence, class_name = detection
                                
                                # Draw bounding box with slightly faded color
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                                
                                # Draw label
                                label = f"{class_name}: {confidence:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), (0, 200, 0), -1)
                                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        out.write(annotated_frame)
                        continue
                    
                    processed_frames += 1
                    
                    # Run YOLO prediction on this frame
                    results = model.predict(
                        source=frame,
                        conf=0.70,  # Confidence threshold
                        iou=0.45,   # NMS IoU threshold
                        verbose=False
                    )
                    
                    # Get the first result (since we're processing one frame)
                    result = results[0]
                    
                    # Draw bounding boxes and labels on frame
                    annotated_frame = frame.copy()
                    current_detections = []
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        frames_without_detection = 0  # Reset counter
                        
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get confidence and class
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Get class name (if available)
                            class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                            
                            # Store detection for future frames
                            current_detections.append((x1, y1, x2, y2, class_id, confidence, class_name))
                            
                            # Draw bounding box with bright color for new detections
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # Save detection to text file
                            txt_file.write(f"{frame_count},{class_id},{confidence:.4f},{x1},{y1},{x2},{y2}\n")
                    else:
                        frames_without_detection += 1
                    
                    # Update previous detections
                    if current_detections:
                        previous_detections = current_detections
                    elif frames_without_detection < max_frames_without_detection and previous_detections:
                        # Use previous detections but with faded appearance
                        for detection in previous_detections:
                            x1, y1, x2, y2, class_id, confidence, class_name = detection
                            
                            # Calculate fade factor based on frames without detection
                            fade_factor = 1.0 - (frames_without_detection / max_frames_without_detection)
                            color_intensity = int(255 * fade_factor)
                            
                            # Draw bounding box with faded color
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, color_intensity, 0), 2)
                            
                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, color_intensity, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    else:
                        # Clear previous detections if too many frames without detection
                        previous_detections = []
                    
                    # Write processed frame to output video
                    out.write(annotated_frame)
                    
                    # Print progress every 100 frames
                    if processed_frames % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps_processed = processed_frames / elapsed_time
                        print(f"   Processed {processed_frames} frames ({fps_processed:.1f} fps)")
                        if previous_detections:
                            print(f"   Active detections: {len(previous_detections)}")
            
            # Clean up
            cap.release()
            out.release()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"✅ Completed: {video_path.name}")
            print(f"   Total frames: {frame_count}")
            print(f"   Processed frames: {processed_frames}")
            print(f"   Processing time: {total_time:.2f} seconds")
            print(f"   Average FPS: {processed_frames/total_time:.1f}")
            print(f"   Output video: {output_video_path}")
            print(f"   Detections file: {output_txt_path}")
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {str(e)}")
            traceback.print_exc()
            continue
    
    print(f"\n🎉 All videos processed! Check the '{output_dir}' folder for results.")

if __name__ == "__main__":
    print("🚀 Starting frame-by-frame video inference with YOLO11...")
    process_videos()

