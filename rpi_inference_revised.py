from gpiozero import LED
from picamera2 import PiCamera

# Initialize GPIO Zero
led_1 = LED(23)
led_2 = LED(24)

# Load model
import torch
import numpy as np

torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

def picamera_image_to_tensor(img):
  # Convert RGB to a batched tensor
  img = np.expand_dims(img, axis=0)
  # Move the color channel to dim 1
  img = img.transpose(0, 3, 1, 2)
  # Convert to torch tensor and set the data type
  return torch.tensor(img, dtype=torch.float32)

try:
  with PiCamera() as camera:
    camera.resolution = (224, 224)

    while True:
      # Capture an image from the camera
      img = camera.capture()

      # Process the image through the model
      inputs = picamera_image_to_tensor(img)
      outputs = model(inputs)
      prediction = outputs[0].max(1)[1].item()

      # Change LEDs based on our prediction
      if prediction == 0:
          led_1.on()
          led_2.off()
      elif prediction == 1:
          led_1.off()
          led_2.on()

except KeyboardInterrupt:
  print("Interrupted, cleaning up...")
finally:
  # Release resources
  led_1.off()
  led_2.off()
