#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import gpiod

# Get GPIO pins for LED's: 23, 24 are pins 16, 18
leds = gpiod.Chip('gpiochip0').get_lines([23,24])
leds.request(
    consumer='pi_ai',
    type=gpiod.LINE_REQ_DIR_OUT,
    default_vals=[1,1]
)

# Set up camera capture
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

# Turn off LEDs
leds.set_values([0,0])

# Load model
torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load('sfdc_tutorial_classifier.pth')
torch.no_grad()

def opencv_image_to_tensor(img):
    # Convert BGR to RGB and add batch channel
    img = img[None,:,:,[2,1,0]]
    # Move the color channel to dim 1
    img = img.transpose(0,3,1,2)
    # Convert to a torch tensor
    return torch.tensor(img)

while True:
    # Get an image from the camera
    ret, image = camera.read()
    assert ret, "Could not read frame"

    # Process the image through the model
    inputs = opencv_image_to_tensor(image)
    outputs = model(inputs)
    prediction = outputs[0].max()[1].item()

    # Change LED's based on our prediction
    if prediction==0:
        leds.set_values([1,0])
    elif prediction==1:
        leds.set_values([0,1])
