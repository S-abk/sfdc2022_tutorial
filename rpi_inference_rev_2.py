#!/usr/bin/env python3
import torch
import numpy as np
import cv2
from gpiozero import LED
from picamera2 import Picamera2

# Initialize LEDs using gpiozero
led1 = LED(23)
led2 = LED(24)

# Turn off LEDs initially
led1.off()
led2.off()

# Initialize Pi Camera
try:
    piCam = Picamera2()
except Exception as e:
    print(f"Failed to initialize Pi Camera: {e}")
    exit(1)

# Configure the preview settings
piCam.preview_configuration.main.size = (224, 224)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

# Load PyTorch model
torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

def opencv_image_to_tensor(img):
    img = img[None, :, :, [2, 1, 0]]
    img = img.transpose(0, 3, 1, 2)
    return torch.tensor(img)

# Main loop for capturing and displaying frames
try:
    while True:
        frame = piCam.capture_array()
        cv2.imshow("piCam", frame)

        # Process the image through the model
        inputs = opencv_image_to_tensor(frame)
        outputs = model(inputs)
        prediction = outputs[0].max()[1].item()

        # Change LEDs based on prediction
        if prediction == 0:
            led1.on()
            led2.off()
        elif prediction == 1:
            led1.off()
            led2.on()

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # Cleanup
    cv2.destroyAllWindows()
    piCam.stop()  # Assuming Picamera2 has a stop() method for releasing resources
    led1.off()
    led2.off()
