import cv2
from ultralytics import YOLO
import time
import torch
import requests

# ----------------------------
# Configuration
# ----------------------------
PI_IP = "192.168.1.104"
PORT = 5000
MODEL_PATH = "yolov8n.onnx"

CONF_THRESHOLD = 0.65
FRAMES_REQUIRED = 5
RESET_FRAMES = 10

PUSHOVER_USER_KEY = "u9t6fwaxb83gva966iv5decmo11i6b"
PUSHOVER_API_TOKEN = "a4rqs8k2avow2zczez4k235sytngdr"
# ----------------------------

torch.set_num_threads(4)

print("Loading ONNX model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

stream_url = f"tcp://{PI_IP}:{PORT}?tcp_nodelay=1"
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open TCP stream")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

consecutive_detections = 0
no_detection_frames = 0
event_active = False
prev_time = time.time()

def send_notification_with_image(frame):
    # Save snapshot
    image_path = "alert.jpg"
    cv2.imwrite(image_path, frame)

    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": "🚨 Person detected!",
        },
        files={
            "attachment": open(image_path, "rb")
        }
    )

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(
        frame,
        imgsz=640,
        classes=[0],
        conf=CONF_THRESHOLD,
        verbose=False
    )

    annotated = results[0].plot()

    # Count valid detections above threshold
    boxes = results[0].boxes
    person_detected = len(boxes) > 0

    if person_detected:
        consecutive_detections += 1
        no_detection_frames = 0
    else:
        consecutive_detections = 0
        no_detection_frames += 1

    # Trigger event
    if consecutive_detections >= FRAMES_REQUIRED and not event_active:
        print("Confirmed person detection. Sending alert...")
        send_notification_with_image(annotated)
        event_active = True

    # Reset event when person leaves
    if no_detection_frames >= RESET_FRAMES:
        event_active = False

    # FPS display
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Smart Security Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
