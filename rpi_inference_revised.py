#!/usr/bin/env python3
import torch
import numpy as np
import cv2
import RPi.GPIO as GPIO

try:
    # Initialize GPIO for LEDs
    LED_PIN1 = 23
    LED_PIN2 = 24
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([LED_PIN1, LED_PIN2], GPIO.OUT)

    # Turn off LEDs initially
    GPIO.output(LED_PIN1, GPIO.LOW)
    GPIO.output(LED_PIN2, GPIO.LOW)

    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Could not open video device")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

    # Load PyTorch model
    torch.backends.quantized.engine = 'qnnpack'
    model = torch.jit.load('sfdc_tutorial_classifier.pth')
    if model is None:
        raise Exception("Could not load the model")
    
    torch.no_grad()

    def opencv_image_to_tensor(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        return torch.from_numpy(img).float()

    while True:
        ret, image = camera.read()
        if not ret:
            raise Exception("Could not read frame")

        inputs = opencv_image_to_tensor(image)
        outputs = model(inputs)
        prediction = outputs[0].max()[1].item()

        # Update LEDs based on prediction
        if prediction == 0:
            GPIO.output(LED_PIN1, GPIO.HIGH)
            GPIO.output(LED_PIN2, GPIO.LOW)
        elif prediction == 1:
            GPIO.output(LED_PIN1, GPIO.LOW)
            GPIO.output(LED_PIN2, GPIO.HIGH)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup GPIO and camera
    if 'camera' in locals():
        camera.release()
    GPIO.cleanup()
