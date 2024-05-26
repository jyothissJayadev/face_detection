# Face Detection and Blur Project

<p align="center">
  <img src="[images/project_screenshot.png](https://cdn-images-1.medium.com/v2/1*9UGEaDhhWVmqEtmVXA394A.jpeg)" alt="Project Screenshot" width="500">
</p>

This project captures video from a webcam, detects faces in real-time using MediaPipe, applies a blur effect to the detected faces, and displays the processed video stream.

## Features

- Real-time face detection using MediaPipe
- Blurs detected faces for privacy
- Displays the video stream with blurred faces

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/face-detection-blur.git
    cd face-detection-blur
    ```

2. Install the required Python packages:
    ```bash
    pip install opencv-python-headless mediapipe
    ```

## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. The webcam feed will open, and faces will be detected and blurred in real-time.

3. Press `q` to quit the video feed.

## Code Overview

### main.py

```python
import cv2
import mediapipe as mp

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Could not open video.")
    exit()

# Initialize the face detection model
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(img_rgb)

    # Check if any faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bo
