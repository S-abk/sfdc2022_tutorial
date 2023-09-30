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
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

# Function to convert OpenCV image to PyTorch tensor
def opencv_image_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    img = img.unsqueeze(0)
    
    # Quantize the tensor to match the model (Assuming qint8 quantization)
    scale = 0.1
    zero_point = 128
    img = torch.quantize_per_tensor(img, scale=scale, zero_point=zero_point, dtype=torch.qint8)

    return img

# Main Loop
while True:
    frame = piCam.capture_array()

    tensor_input = opencv_image_to_tensor(frame)
    
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
