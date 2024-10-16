!pip install ultralytics
!pip install tensorflow opencv-python

import ultralytics
from ultralytics import YOLO
import os
import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Load the YOLO model
model = YOLO('yolov8l-world.pt')
model.set_classes(["helmet", "gloves", "glasses", "masks"])

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    detections_per_frame = []  # This will hold the YOLO detections for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO prediction on each frame
        results = model.predict(frame, device='cpu')

        # Collect detections for this frame
        boxes, labels = [], []
        for result in results:
            for det in result.boxes:
                x1, y1, x2, y2 = det.xyxy[0].numpy()  # bounding box coordinates
                label = int(det.cls[0])  # detected class index
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
        # Append frame's detections (e.g., box coordinates + class labels)
        detections_per_frame.append((boxes, labels))

    cap.release()
    return detections_per_frame

# Function to process uploaded image
def process_image(image_path):
    image = cv2.imread(image_path)
    
    # YOLO prediction
    results = model.predict(image, device='cpu')

    # Display the results
    results[0].show()  # Show bounding boxes and labels

# Main function to choose between video or image input
def main():
    choice = input("Type 'video' to process a video or 'image' to upload an image: ").strip().lower()
    
    if choice == 'video':
        video_path = input("Enter the path to the video file: ")
        detections = process_video(video_path)
        print(f"Processed {len(detections)} frames from the video.")
        
    elif choice == 'image':
        # Upload image files to Colab and detect objects
        uploaded = files.upload()
        # Assuming one image file is uploaded, read and process it
        for image_name in uploaded.keys():
            process_image(image_name)
    else:
        print("Invalid choice. Please type 'video' or 'image'.")

# Run the main function
main()
