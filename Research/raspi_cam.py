from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_video_configuration({"size": (1920, 1080) ,"format": "RGB888"})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    print(type(frame), frame.shape)
