import cv2
from time import monotonic

from src.camera import Camera
from src.detector import YOLODetector
from src.tracker import CentroidTracker

from src.zones import build_default_zones, compute_zone_hits
from src.events import EventEngine
from src.zones import polygon_norm_to_px
from src.notifier import Notifier, Notification


def main():
    # Initialize camera, detector, and tracker
    cam = Camera()
    detector = YOLODetector()
    tracker = CentroidTracker(
        max_disappeared=15,
        max_distance=60
    )

    # ---- Day 6: zones + events ----
    zones = build_default_zones()
    event_engine = EventEngine(
        loiter_threshold_s=5.0,   # short for testing
        exit_debounce_s=0.2,
    )
    notifier = Notifier(
        cooldown_s=30.0  # short cooldown for testing
    )

    while True:
        data = cam.read()
        if data is None:
            break

        # Camera.read() may return (frame, metadata)
        if isinstance(data, tuple):
            frame = data[0]
        else:
            frame = data

        # Run detection
        detections = detector.detect(frame)

        # Extract bounding boxes only for tracker
        boxes = [box for (box, conf) in detections]

        # Update tracker
        objects = tracker.update(boxes)  # {id: (cx, cy)}

        # ---- Draw bounding boxes ----
        for (x, y, w, h), conf in detections:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ---- Draw tracked IDs ----
        for object_id, (cx, cy) in objects.items():
            cv2.putText(
                frame,
                f"ID {object_id}",
                (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

        # ---- Day 6: zone hits + events ----
        h, w = frame.shape[:2]
        # ---- Draw zones ----
        for z in zones:
            pts = polygon_norm_to_px(z.polygon_norm, w, h)
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Label zone near first vertex
            label_x, label_y = pts[0]
            cv2.putText(
                frame,
                z.name,
                (label_x + 5, label_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        t_now = monotonic()

        zone_hits = compute_zone_hits(
            objects_px=objects,
            w=w,
            h=h,
            zones=zones,
        )

        events = event_engine.update(zone_hits, t_now)
        for e in events:
            print(
                f"[EVENT] t={e.t:.2f} "
                f"{e.type.upper()} "
                f"zone={e.zone} "
                f"id={e.obj_id} "
                f"meta={e.meta}"
            )
            # ---- Notification trigger ----
            if e.type == "enter":
                notification = Notification(
                    t=e.t,
                    message=f"Person entered {e.zone}",
                    zone=e.zone,
                    obj_id=e.obj_id,
                )
                notifier.send(notification)
        # ---- Display output ----
        cv2.imshow("Autonomous Vision Node", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

