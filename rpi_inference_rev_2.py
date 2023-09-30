#!/usr/bin/env python3
import torch
import cv2
from gpiozero import LED
from picamera2 import Picamera2
import numpy as np

# Initialize LEDs
led1 = LED(23)
led2 = LED(24)

# Turn off LEDs
led1.off()
led2.off()

# Initialize Pi Camera
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

# Load PyTorch model
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

def opencv_image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # Convert to float
    img = torch.tensor(img).permute(2, 0, 1)  # C, H, W
    img = img.unsqueeze(0)  # N, C, H, W
    return img

# Main loop for capturing and displaying frames
try:
    while True:
        frame = piCam.capture_array()

        # Prepare and make prediction
        inputs = opencv_image_to_tensor(frame)
        outputs = model(inputs)
        prediction = outputs[0].max()[1].item()

        # Control LEDs based on prediction
        if prediction == 0:
            led1.on()
            led2.off()
        elif prediction == 1:
            led1.off()
            led2.on()

        # Display the frame
        cv2.imshow("piCam", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # Clean-up
    cv2.destroyAllWindows()
    piCam.stop()  # Assuming Picamera2 has a stop() method
    led1.off()
    led2.off()
