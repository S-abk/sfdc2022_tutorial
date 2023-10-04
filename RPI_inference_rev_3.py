import cv2
import torch
import numpy as np
from gpiozero import LED
from picamera2 import Picamera2

def initialize_leds():
    """Initialize the LEDs and return them."""
    led1 = LED(23)
    led2 = LED(24)
    led1.off()
    led2.off()
    return led1, led2

def initialize_camera():
    """Initialize the Pi Camera and return it."""
    piCam = Picamera2()
    piCam.preview_configuration.main.size = (224, 224)
    piCam.preview_configuration.main.format = "RGB888"
    piCam.preview_configuration.align()
    piCam.configure("preview")
    piCam.start()
    return piCam

def load_model(model_path):
    """Load PyTorch model from a given path."""
    return torch.jit.load(model_path)

def opencv_image_to_tensor(img):
    """Convert OpenCV image to PyTorch tensor."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    img = img.unsqueeze(0)
    
    # Quantize the tensor to match the model (Assuming qint8 quantization)
    scale = 0.1
    zero_point = 128
    img = torch.quantize_per_tensor(img, scale=scale, zero_point=zero_point, dtype=torch.qint8)

    return img

def main_loop(piCam, model, led1, led2):
    """Capture frames, make predictions, and control LEDs."""
    while True:
        frame = piCam.capture_array()

        tensor_input = opencv_image_to_tensor(frame)

        with torch.no_grad():
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

def cleanup(piCam, led1, led2):
    """Release resources."""
    cv2.destroyAllWindows()
    piCam.stop()
    led1.off()
    led2.off()

if __name__ == "__main__":
    led1, led2 = initialize_leds()
    piCam = initialize_camera()
    model = load_model('sfdc_tutorial_classifier.pth')
    main_loop(piCam, model, led1, led2)
    cleanup(piCam, led1, led2)
