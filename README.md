# AI-Helmet-ATM-Safety
Helmet detection in ATM'S and playing alarm to remove the helmet

import cv2
import numpy as np
from IPython.display import display, Image, Audio
import time

# Define paths to YOLOv3 model and configuration files
weights_path = '/content/drive/MyDrive/DLITHE/helmet/helmet/yolov3-helmet.weights'
config_path = '/content/drive/MyDrive/DLITHE/helmet/helmet/yolov3-helmet.cfg'
names_path = '/content/drive/MyDrive/DLITHE/helmet/helmet/helmet.names'  # File containing class names

# Load YOLOv3 model
net = cv2.dnn.readNet(weights_path, config_path)

# Load class names for helmet detection
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Define the path to your video file
video_path = '/content/drive/MyDrive/DLITHE/helmet/INPUT FILES/pexels-c-technical-5803206 (2160p).mp4'

# Initialize video capture from the specified file
cap = cv2.VideoCapture(video_path)

# Define the buzzer sound
buzzer_sound_path = '/content/drive/MyDrive/DLITHE/helmet/INPUT FILES/REMOVE YOUR HELMET.mp3'

# Define the interval for capturing frames (10 seconds)
frame_capture_interval = 1  # in seconds

# Initialize the timer
start_time = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    if elapsed_time >= frame_capture_interval:
        # Reset the timer
        start_time = time.time()

        # Create a blob from the input frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Perform forward pass through the network
        outs = net.forward(output_layer_names)

        # Initialize lists for bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Set confidence threshold for detection
        conf_threshold = 0.5

        # Set non-maximum suppression threshold
        nms_threshold = 0.4

        # Iterate over each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold and classes[class_id] == "Helmet":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw bounding boxes and labels on the frame
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Play the buzzer sound when a helmet is detected
                display(Audio(buzzer_sound_path, autoplay=True))

        # Convert the frame to RGB format for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detections
        display(Image(data=cv2.imencode('.jpg', frame_rgb)[1]))

# Release the video file
cap.release()
