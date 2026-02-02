import cv2
import time

class Camera:
    def __init__(self, width=640, height=480):
        self.pipeline = (
            "libcamerasrc ! "
            f"video/x-raw,width={width},height={height},format=RGB ! "
            "videoconvert ! appsink drop=1"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Camera could not be opened")

        self.last_time = time.time()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0

        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        return frame, fps

    def release(self):
        self.cap.release()
