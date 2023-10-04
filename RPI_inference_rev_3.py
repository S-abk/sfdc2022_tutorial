import cv2
import torch
from gpiozero import LED
from picamera2 import Picamera2

# Setup
torch.backends.quantized.engine = 'qnnpack'

# Initialization
led1 = LED(23)
led2 = LED(24)
led1.off()import cv2
import torch
from gpiozero import LED
from picamera2 import PiCamera2  # Ensure correct spelling

# Setup: Set the quantized engine
torch.backends.quantized.engine = 'qnnpack'

# Initialization: Initialize LEDs and Camera
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

# Load the trained model
model = torch.jit.load('sfdc_tutorial_classifier.pth')

def image_to_tensor(img):
    """
    Convert the captured frame to a tensor.
    1. Add a batch channel.
    2. Convert BGR to RGB (if the image is captured in BGR).
    3. Rearrange dimensions to match the input format expected by PyTorch.
    4. Convert the image data to a tensor of type float32.
    """
    img = img[None, :, :, [2, 1, 0]]  # Assuming img is in BGR format
    img = img.transpose(0, 3, 1, 2)  # Rearrange dimensions
    img = torch.tensor(img, dtype=torch.float32)  # Convert to torch tensor
    return img

# Main inference loop
try:
    while True:
        # Capture a frame
        frame = piCam.capture_array()

        # Convert the captured frame to a tensor
        tensor_input = image_to_tensor(frame)

        # Run inference with no gradient computation
        with torch.no_grad():
            output = model(tensor_input)

        # Get the predicted class index
        prediction = output[0].max(0)[1].item()

        # Act on the prediction by controlling LEDs
        if prediction == 0:
            led1.on()
            led2.off()
        elif prediction == 1:
            led1.off()
            led2.on()

        # Display the frame
        cv2.imshow("Real-Time Prediction", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup: Release resources
    cv2.destroyAllWindows()
    piCam.stop()
    led1.off()
    led2.off()

led2.off()

piCam = Picamera2()
piCam.preview_configuration.main.size = (224, 224)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

#def image_to_tensor(img):
#    """Convert the captured frame to a tensor."""
#    img = torch.tensor(img).permute(2, 0, 1)
#    img = img.unsqueeze(0)
#    return img

#def image_to_tensor(img):
#    """Convert the captured frame to a tensor."""
#    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Specify dtype as float32
#    img = img.unsqueeze(0)
#    return img


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
