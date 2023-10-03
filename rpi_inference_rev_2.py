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
piCam = Picamera2()
piCam.preview_configuration.main.size = (224, 224)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

# Load the PyTorch model
torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

# Function to convert image to PyTorch tensor
def image_to_tensor(img):
    img = torch.tensor(img).permute(2, 0, 1)
    img = img.unsqueeze(0)
    return img

# Main Loop
while True:
    frame = piCam.capture_array()

    tensor_input = image_to_tensor(frame)
    
    try:
        output = model(tensor_input)
        prediction = output[0].max(0)[1].item()
    except Exception as e:
        print(f"Model inference failed: {e}")
        continue

    if prediction == 0:
        led1.on()
        led2.off()
    elif prediction == 1:
        led1.off()
        led2.on()

    cv2.imshow("Real-Time Prediction", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
piCam.stop()
led1.off()
led2.off()
