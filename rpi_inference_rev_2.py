#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from gpiozero import LED
from picamera2 import Picamera2

# Initialize the LEDs
led1 = LED(23)
led2 = LED(24)

# Turn off LEDs initially
led1.off()
led2.off()

# Initialize the Pi Camera
try:
    piCam = Picamera2()
except Exception as e:
    print(f"Failed to initialize Pi Camera: {e}")
    exit(1)

# Configure camera settings
piCam.preview_configuration.main.size = (224, 224)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

# Load the PyTorch model
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

# Function to convert OpenCV image to PyTorch tensor
def opencv_image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    img = torch.tensor(img).permute(2, 0, 1)  # C, H, W
    img = img.unsqueeze(0)  # N, C, H, W
    return img

# Main Loop
try:
    while True:
        # Capture a frame
        frame = piCam.capture_array()
        
        # Convert frame to PyTorch tensor and make a prediction
        tensor_input = opencv_image_to_tensor(frame)
        output = model(tensor_input)
        prediction = output[0].max(0)[1].item()
        
        # Control LEDs based on the prediction
        if prediction == 0:
            led1.on()
            led2.off()
        elif prediction == 1:
            led1.off()
            led2.on()
        
        # Display the frame
        cv2.imshow("Real-Time Prediction", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    cv2.destroyAllWindows()
    piCam.stop()  # Assuming Picamera2 has a stop() method for releasing resources
    led1.off()
    led2.off()
