import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define your class indices and names
class_names = ['Red', 'Green', 'Blue', 'Black', 'White']

# Load your pre-trained model
model = load_model('/Users/esmaelmoataz/Documents/color_detection_cnn_compatible.h5')

# Initialize the video capture object for the camera
cap = cv2.VideoCapture(0)  # 0 means the default camera

# Define the size of the region of interest (ROI)
roi_size = (64, 64)  # Adjust as needed based on phone size

# Define color ranges
color_ranges = {
    'Red': ([0, 120, 70], [10, 255, 255]),
    'Green': ([36, 100, 100], [86, 255, 255]),
    'Blue': ([94, 80, 2], [126, 255, 255]),
    'Black': ([0, 0, 0], [180, 255, 30]),
    'White': ([0, 0, 168], [172, 111, 255])
}

# Function to preprocess an image
def preprocess_image(image):
    resized = cv2.resize(image, roi_size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Function to detect the color of an ROI
def detect_color(roi):
    input_data = preprocess_image(roi)
    predictions = model.predict(input_data)
    color_class_index = np.argmax(predictions)
    detected_color_name = class_names[color_class_index]
    confidence_score = np.max(predictions) * 100
    return detected_color_name, confidence_score

# Function to draw bounding boxes and label detected colors
def draw_results(frame, results):
    for i, (color_name, confidence_score) in enumerate(results):
        x, y = i * roi_size[0], i * roi_size[1]
        cv2.rectangle(frame, (x, y), (x + roi_size[0], y + roi_size[1]), (0, 255, 0), 2)
        cv2.putText(frame, f'{color_name} ({confidence_score:.2f}%)',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Resize the frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Define ROIs based on phone positions (adjust positions as needed)
    rois = [
        frame_resized[0:roi_size[1], 0:roi_size[0]],
        frame_resized[0:roi_size[1], frame_resized.shape[1] - roi_size[0]:frame_resized.shape[1]],
        frame_resized[frame_resized.shape[0] - roi_size[1]:frame_resized.shape[0], 0:roi_size[0]],
        frame_resized[frame_resized.shape[0] - roi_size[1]:frame_resized.shape[0], frame_resized.shape[1] - roi_size[0]:frame_resized.shape[1]]
    ]

    # Detect colors for each ROI
    results = [detect_color(roi) for roi in rois]

    # Draw bounding boxes and labels
    draw_results(frame_resized, results)

    # Display the frame with the detected objects
    cv2.imshow('Color Detection Output', frame_resized)

    # Exit on pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()