import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from collections import deque
import mediapipe as mp

# Define the path to your model file
model_path = '/Users/esmaelmoataz/Documents/Machine learning/model_checkpoints/model_epoch_15.pth'

# Define your ResNet10 model
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet10(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet10, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Load the trained model
model = ResNet10(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Parameters
frame_height = 128
frame_width = 128
gesture_names = ['dislike', 'like', 'stop']
gesture_confidence_thresholds = {'dislike': 0.5, 'like': 0.5, 'stop': 0.5}
consecutive_predictions = 10

# Preprocessing function
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((frame_height, frame_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    image = data_transforms(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to make a prediction on a single frame
def predict_single_frame(frame):
    frame = frame.to(device)
    with torch.no_grad():
        outputs = model(frame)
    _, predicted = torch.max(outputs, 1)
    predicted_gesture = gesture_names[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
    return predicted_gesture, confidence, outputs.cpu().numpy()

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)

def detect_hands(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results.multi_hand_landmarks is not None

# Function to capture and predict gestures in real-time
def predict_gestures_in_real_time():
    cap = cv2.VideoCapture(0)
    confirmed_gesture = None
    confidence_deque = deque(maxlen=consecutive_predictions)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time gesture prediction...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        if detect_hands(frame):
            processed_frame = preprocess_frame(frame)
            predicted_gesture, confidence, all_confidences = predict_single_frame(processed_frame)

            confidence_deque.append((predicted_gesture, confidence))

            if len(confidence_deque) == consecutive_predictions:
                gestures, confidences = zip(*confidence_deque)
                if all(g == gestures[0] for g in gestures) and all(
                        c >= gesture_confidence_thresholds[gestures[0]] for c in confidences):
                    confirmed_gesture = gestures[0]
                else:
                    confirmed_gesture = None

            y_offset = 30
            for i, gesture in enumerate(gesture_names):
                confidence_text = f'{all_confidences[0][i]:.2f}' if i < len(all_confidences[0]) else 'N/A'
                text = f'{gesture}: {confidence_text}'
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 30

            if confirmed_gesture:
                cv2.putText(frame, f'Confirmed Gesture: {confirmed_gesture}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped real-time gesture prediction.")

# Example usage
predict_gestures_in_real_time()