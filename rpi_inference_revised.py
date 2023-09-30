import torch
import numpy as np
import picamera
import picamera.array
import RPi.GPIO as GPIO
import time

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
LED_PIN_1 = 23
LED_PIN_2 = 24
GPIO.setup(LED_PIN_1, GPIO.OUT)
GPIO.setup(LED_PIN_2, GPIO.OUT)

# Turn off LEDs initially
GPIO.output(LED_PIN_1, GPIO.LOW)
GPIO.output(LED_PIN_2, GPIO.LOW)

# Load model
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
    with picamera.PiCamera() as camera:
        camera.resolution = (224, 224)
        with picamera.array.PiRGBArray(camera) as stream:
            while True:
                camera.capture(stream, 'rgb')
                image = stream.array

                # Process the image through the model
                inputs = picamera_image_to_tensor(image)
                outputs = model(inputs)
                prediction = outputs[0].max(1)[1].item()

                # Change LEDs based on our prediction
                if prediction == 0:
                    GPIO.output(LED_PIN_1, GPIO.HIGH)
                    GPIO.output(LED_PIN_2, GPIO.LOW)
                elif prediction == 1:
                    GPIO.output(LED_PIN_1, GPIO.LOW)
                    GPIO.output(LED_PIN_2, GPIO.HIGH)

                # Clear the stream for the next capture
                stream.truncate(0)

except KeyboardInterrupt:
    print("Interrupted, cleaning up...")
finally:
    # Release resources
    GPIO.cleanup()
