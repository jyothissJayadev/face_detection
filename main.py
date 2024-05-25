import cv2
import mediapipe as mp

# Initialize video capture
video = cv2.VideoCapture(1)
if not video.isOpened():
    print("Could not open video.")
    exit()

# Initialize the face detection model
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def mainFunction(img_rgb):
    output = face_detection.process(img_rgb)
    if output.detections:
        for detection in output.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
            x1, y1 = int(x * img_rgb.shape[1]), int(y * img_rgb.shape[0])
            x2, y2 = int((x + w) * img_rgb.shape[1]), int((y + h) * img_rgb.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_rgb

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    img_rgb = mainFunction(img_rgb)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
