import cv2
import torch
from gpiozero import LED
from picamera2 import PiCamera2
from picamera2 import ControlId

# Setup
torch.backends.quantized.engine = 'qnnpack'

# Initialization
led1 = LED(23)
led2 = LED(24)
led1.off()
led2.off()

piCam = PiCamera2()
piCam.preview_configuration.main.size = (224, 224)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

def image_to_tensor(img):
    """Convert the captured frame to a tensor."""
    img = img[None, :, :, [2, 1, 0]]  # Add a batch channel and convert BGR to RGB
    img = img.transpose(0, 3, 1, 2)  # Move the color channel to dim 1
    img = torch.tensor(img, dtype=torch.float32)  # Convert to a torch tensor with float32 type
    return img

# Main inference loop
try:
    while True:
        frame = piCam.capture_array()
        tensor_input = image_to_tensor(frame)

        output = model(tensor_input)
        prediction = output[0].max(0)[1].item()

        if prediction == 0:
            led1.on()
            led2.off()
        elif prediction == 1:
            led1.off()
            led2.on()

        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    piCam.stop()
    led1.off()
    led2.off()
