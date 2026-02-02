import cv2
import numpy as np


class YOLODetector:
    def __init__(self,
                 model_path="models/yolov8n.onnx",
                 conf_threshold=0.15,
                 iou_threshold=0.5):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_size = 640

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (self.input_size, self.input_size),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        preds = self.net.forward()

        preds = np.squeeze(preds)  # shape: (84, 8400) or (8400, 84)
        if preds.shape[0] == 84:
            preds = preds.T  # (8400, 84)

        boxes = []
        confidences = []

        for pred in preds:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # COCO person class = 0
            if class_id != 0 or confidence < self.conf_threshold:
                continue

            cx, cy, bw, bh = pred[0:4]

            x = int((cx - bw / 2) * w / self.input_size)
            y = int((cy - bh / 2) * h / self.input_size)
            bw = int(bw * w / self.input_size)
            bh = int(bh * h / self.input_size)

            boxes.append([x, y, bw, bh])
            confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.iou_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append((boxes[i], confidences[i]))

        return detections

