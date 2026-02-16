import logging
import os
import queue
import signal
import threading
import time

import cv2
import requests
import torch
from ultralytics import YOLO


CONFIG_WARNINGS = []


def load_dotenv_file(path=".env"):
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logging.warning("Could not read .env file %s: %s", path, exc)


load_dotenv_file(".env")


def get_env_str(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_env_int(name, default, min_value=None, max_value=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        CONFIG_WARNINGS.append(f"{name}={raw!r} is invalid. Using default {default}.")
        return default

    if min_value is not None and value < min_value:
        CONFIG_WARNINGS.append(f"{name}={value} is below minimum {min_value}. Using {min_value}.")
        value = min_value
    if max_value is not None and value > max_value:
        CONFIG_WARNINGS.append(f"{name}={value} is above maximum {max_value}. Using {max_value}.")
        value = max_value
    return value


def get_env_float(name, default, min_value=None, max_value=None):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        CONFIG_WARNINGS.append(f"{name}={raw!r} is invalid. Using default {default}.")
        return default

    if min_value is not None and value < min_value:
        CONFIG_WARNINGS.append(f"{name}={value} is below minimum {min_value}. Using {min_value}.")
        value = min_value
    if max_value is not None and value > max_value:
        CONFIG_WARNINGS.append(f"{name}={value} is above maximum {max_value}. Using {max_value}.")
        value = max_value
    return value


def get_env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    CONFIG_WARNINGS.append(f"{name}={raw!r} is invalid. Using default {default}.")
    return default


# ----------------------------
# Configuration (env overridable)
# ----------------------------
MODEL_PATH = get_env_str("MODEL_PATH", "yolov8n.onnx")

IMG_SIZE = get_env_int("IMG_SIZE", 416, min_value=64, max_value=1280)
INFER_EVERY_N = get_env_int("INFER_EVERY_N", 2, min_value=1, max_value=60)
CONF_THRESHOLD = get_env_float("CONF_THRESHOLD", 0.65, min_value=0.0, max_value=1.0)
FRAMES_REQUIRED = get_env_int("FRAMES_REQUIRED", 5, min_value=1, max_value=300)
RESET_FRAMES = get_env_int("RESET_FRAMES", 10, min_value=1, max_value=300)

TORCH_THREADS = get_env_int("TORCH_THREADS", 4, min_value=1, max_value=64)
DISPLAY_WINDOW = get_env_bool("DISPLAY_WINDOW", True)
WINDOW_NAME = get_env_str("WINDOW_NAME", "Smart Security Detection Pi")

READ_FAIL_REOPEN_THRESHOLD = get_env_int("READ_FAIL_REOPEN_THRESHOLD", 25, min_value=1, max_value=10000)
READ_RETRY_SLEEP_SEC = get_env_float("READ_RETRY_SLEEP_SEC", 0.02, min_value=0.0, max_value=10.0)
RECONNECT_BACKOFF_MAX_SEC = get_env_float("RECONNECT_BACKOFF_MAX_SEC", 5.0, min_value=0.1, max_value=60.0)

PUSHOVER_USER_KEY = get_env_str("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = get_env_str("PUSHOVER_API_TOKEN", "")
PUSHOVER_TIMEOUT_SEC = get_env_float("PUSHOVER_TIMEOUT_SEC", 5.0, min_value=0.1, max_value=60.0)
ALERT_IMAGE_PATH = get_env_str("ALERT_IMAGE_PATH", "alert.jpg")

CAMERA_MODE = get_env_str("CAMERA_MODE", "libcamera").lower()
CAMERA_INDEX = get_env_int("CAMERA_INDEX", 0, min_value=0, max_value=16)
CAMERA_WIDTH = get_env_int("CAMERA_WIDTH", 640, min_value=64, max_value=4096)
CAMERA_HEIGHT = get_env_int("CAMERA_HEIGHT", 480, min_value=64, max_value=4096)
CAMERA_FPS = get_env_int("CAMERA_FPS", 30, min_value=1, max_value=240)
LIBCAMERA_FLIP_METHOD = get_env_int("LIBCAMERA_FLIP_METHOD", 0, min_value=0, max_value=7)
LIBCAMERA_EXTRA = get_env_str("LIBCAMERA_EXTRA", "")


class LatestFrameCamera:
    def __init__(self):
        self.cap = None
        self.cap_lock = threading.Lock()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.running = False
        self.thread = None

    def _build_capture(self):
        if CAMERA_MODE == "libcamera":
            # For Raspberry Pi OS with libcamera + GStreamer.
            extra = f" {LIBCAMERA_EXTRA.strip()}" if LIBCAMERA_EXTRA.strip() else ""
            gst = (
                "libcamerasrc ! "
                f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! "
                f"videoflip method={LIBCAMERA_FLIP_METHOD} ! "
                "videoconvert ! appsink drop=true max-buffers=1 sync=false"
                f"{extra}"
            )
            return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER), f"gstreamer(libcamera): {gst}"

        if CAMERA_MODE == "v4l2":
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            return cap, f"v4l2 index {CAMERA_INDEX}"

        CONFIG_WARNINGS.append(
            f"CAMERA_MODE={CAMERA_MODE!r} is unsupported. Falling back to v4l2."
        )
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        return cap, f"v4l2 index {CAMERA_INDEX}"

    def _open_capture(self):
        cap, source_desc = self._build_capture()
        if not cap.isOpened():
            cap.release()
            return None, source_desc
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, source_desc

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None
            return self.latest_frame_id, self.latest_frame.copy()

    def _reader_loop(self):
        backoff = 0.1
        read_fail_count = 0

        while self.running:
            with self.cap_lock:
                cap = self.cap

            if cap is None:
                new_cap, source_desc = self._open_capture()
                if new_cap is None:
                    logging.warning("Camera open failed (%s). Retrying in %.2fs", source_desc, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_SEC)
                    continue
                with self.cap_lock:
                    self.cap = new_cap
                logging.info("Connected to camera: %s", source_desc)
                backoff = 0.1
                read_fail_count = 0
                continue

            ret, frame = cap.read()
            if not ret:
                read_fail_count += 1
                if read_fail_count >= READ_FAIL_REOPEN_THRESHOLD:
                    logging.warning("Too many read failures (%d). Reopening camera...", read_fail_count)
                    with self.cap_lock:
                        if self.cap is cap:
                            self.cap.release()
                            self.cap = None
                time.sleep(READ_RETRY_SLEEP_SEC)
                continue

            read_fail_count = 0
            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1


class AlertWorker:
    def __init__(self, token, user_key, timeout_sec, image_path):
        self.token = token
        self.user_key = user_key
        self.timeout_sec = timeout_sec
        self.image_path = image_path
        self.enabled = bool(token and user_key)

        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.session = requests.Session()

    def start(self):
        if not self.enabled:
            logging.warning("Pushover keys missing. Alerts are disabled.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.session.close()

    def enqueue(self, frame, message):
        if not self.enabled:
            return

        try:
            if self.queue.full():
                _ = self.queue.get_nowait()
            self.queue.put_nowait((frame.copy(), message))
        except queue.Empty:
            pass
        except queue.Full:
            logging.warning("Alert queue full; dropping alert frame.")

    def _send_alert(self, frame, message):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            logging.error("Failed to encode alert frame as JPEG.")
            return

        image_bytes = encoded.tobytes()

        try:
            with open(self.image_path, "wb") as f:
                f.write(image_bytes)
        except OSError as exc:
            logging.warning("Could not write snapshot to %s: %s", self.image_path, exc)

        files = {
            "attachment": ("alert.jpg", image_bytes, "image/jpeg")
        }
        data = {
            "token": self.token,
            "user": self.user_key,
            "message": message,
        }

        retries = 2
        for attempt in range(1, retries + 2):
            try:
                resp = self.session.post(
                    "https://api.pushover.net/1/messages.json",
                    data=data,
                    files=files,
                    timeout=(self.timeout_sec, self.timeout_sec),
                )
                resp.raise_for_status()
                logging.info("Alert sent successfully.")
                return
            except requests.RequestException as exc:
                if attempt <= retries:
                    delay = 0.5 * attempt
                    logging.warning("Alert send failed (%s). Retrying in %.1fs...", exc, delay)
                    time.sleep(delay)
                else:
                    logging.error("Alert send failed after retries: %s", exc)

    def _loop(self):
        while self.running:
            try:
                frame, message = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._send_alert(frame, message)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    for warning_msg in CONFIG_WARNINGS:
        logging.warning("Config: %s", warning_msg)

    torch.set_num_threads(TORCH_THREADS)

    logging.info("Loading model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    logging.info("Model loaded.")

    camera = LatestFrameCamera()
    camera.start()

    alert_worker = AlertWorker(
        token=PUSHOVER_API_TOKEN,
        user_key=PUSHOVER_USER_KEY,
        timeout_sec=PUSHOVER_TIMEOUT_SEC,
        image_path=ALERT_IMAGE_PATH,
    )
    alert_worker.start()

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logging.info("Received signal %s. Shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    consecutive_detections = 0
    no_detection_frames = 0
    event_active = False

    prev_time = time.time()
    prev_infer_time = None
    processed_frame_count = 0
    infer_fps = 0.0
    last_camera_frame_id = -1
    last_detections = []

    try:
        while not stop_event.is_set():
            frame_id, frame = camera.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            if frame_id == last_camera_frame_id:
                time.sleep(0.001)
                continue

            last_camera_frame_id = frame_id
            processed_frame_count += 1

            do_infer = (processed_frame_count % INFER_EVERY_N == 0)
            if do_infer:
                infer_start = time.time()
                results = model(
                    frame,
                    imgsz=IMG_SIZE,
                    classes=[0],
                    conf=CONF_THRESHOLD,
                    verbose=False,
                )
                boxes = results[0].boxes

                detections = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf))

                last_detections = detections
                person_detected = len(detections) > 0

                if person_detected:
                    consecutive_detections += 1
                    no_detection_frames = 0
                else:
                    consecutive_detections = 0
                    no_detection_frames += 1

                if consecutive_detections >= FRAMES_REQUIRED and not event_active:
                    logging.info("Confirmed person detection. Queueing alert...")
                    alert_worker.enqueue(frame, "Person detected!")
                    event_active = True

                if no_detection_frames >= RESET_FRAMES:
                    event_active = False

                if prev_infer_time is None:
                    prev_infer_time = infer_start
                else:
                    infer_fps = 1.0 / max(infer_start - prev_infer_time, 1e-6)
                    prev_infer_time = infer_start

            annotated = frame.copy()
            for x1, y1, x2, y2, conf in last_detections:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Person {conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            status_text = "ALERT ACTIVE" if event_active else "ARMED"
            cv2.putText(
                annotated,
                f"Display FPS: {fps:.1f} | Infer FPS: {infer_fps:.1f} | N={INFER_EVERY_N} | {status_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if DISPLAY_WINDOW:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        camera.stop()
        alert_worker.stop()
        if DISPLAY_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
