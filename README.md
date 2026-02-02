# Autonomous Vision Node (Raspberry Pi)

This project is a real-time computer vision system running on a Raspberry Pi.  
It uses a connected camera and a YOLO-based object detection model to perform live inference at ~30 FPS.

The goal of this project is to build a clean, modular, and reproducible edge-vision pipeline suitable for robotics, autonomy, or embedded perception applications.

---

## Features

- Real-time camera capture using OpenCV
- YOLO object detection via Ultralytics
- Live bounding box visualization
- Modular Python package structure
- Runs entirely on Raspberry Pi hardware
- Baseline performance: ~28–30 FPS

---

## Hardware

- Raspberry Pi 4
- Raspberry Pi camera module (or USB camera)
- Power supply
- Internet connection (only required for setup / Git)

---

## Software Stack

- Python 3
- OpenCV
- Ultralytics YOLO
- NumPy

---

## Project Structure

.
├── src/
│ ├── main.py # Entry point (run this)
│ ├── camera.py # Camera interface
│ ├── detector.py # YOLO inference logic
│ ├── visualization.py
│ └── config.py # Tunable parameters
│
├── models/
│ └── yolov8n.pt # YOLO model weights
│
├── camera_test.py # Standalone camera validation script
├── requirements.txt
└── README.md


---

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd autonomous-node
Install dependencies:

pip install -r requirements.txt
Ensure the camera is connected and enabled.

Running the System
From the project root:

python3 -m src.main
A live camera window should open with detected objects outlined by bounding boxes.

Versioning
v0.1-working
First verified working version of the real-time vision pipeline.

Known Limitations
Camera field-of-view is limited by hardware optics

Performance depends on lighting conditions

Model accuracy limited by lightweight YOLO architecture

No object tracking or temporal smoothing yet

Future Work
Object tracking (ID persistence)

Event-based detection (not frame-by-frame)

Region-of-interest filtering

Model optimization for edge performance

Hardware acceleration (NPU / TPU)

Purpose
This project is designed as a foundational perception node for larger autonomous or robotic systems, with an emphasis on clean structure, real-time performance, and extensibility.
