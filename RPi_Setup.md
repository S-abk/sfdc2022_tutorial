# Setting up the Pi

## Prerequisites

At this point, you should have followed the [notebook](SoFloDevCon_Tutorial_2022.ipynb) to train and download your model.

## Hardware Setup

1. Attach your camera module to your Pi.
2. Connect two LED's to pins 16 and 17.
    * Make sure to use appropriate resistors.
4. Insert an SD card with a 64-bit OS.

## Software Setup

1. Install the required packages:
    * `sudo apt install python3, python3-pip, python3-libgpiod, python3-opencv`
    * `python3 -m pip install torch`
2. Place the inference code, "rpi_inference.py" somewhere on your Pi.
3. Get the compiled model file onto your Pi.
    * The inference code expects the model to be called "sfdc_tutorial_classifier.pth" and to be located in the same directory as your script.

## Running

To start inference, run `python3 rpi_inference.py`.

Optionally, you can make the script run on bootup by adding a cron job.
Run `sudo crontab -e` and add the line `@reboot /path/to/rpi_inference.py`.

## That's it!

Congradulations! You have completed the tutorial. You should now have a working image classifier on your Pi.

If it doesn't work, though, don't get discouraged. You often need to play around with the parameters and try it multiple times.

With a bit of time and effort, you can have a bright future in AI!
