import cv2
import time

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera not available")
else:
    print("Camera opened")
    for i in range(5):
        success, frame = camera.read()
        if success:
            print(f"Frame {i}: shape {frame.shape}, mean {frame.mean()}")
        else:
            print(f"Frame {i}: failed")
        time.sleep(1)
    camera.release()