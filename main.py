import cv2
import time
import threading
import queue
import torch
import importlib
import os
import sys
import logging
import psutil
import signal
import numpy as np

from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
import paho.mqtt.client as mqtt
from camera_registry import CAMERAS

# ================== PATH FIX ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ================= CONFIG =================
FRAME_WIDTH        = 480
FRAME_HEIGHT       = 270
CONFIDENCE         = 0.25
IMG_SIZE           = 640
SKIP_FRAMES        = 2
AUTO_OFF_SECONDS   = 6
SEGMENT_OFF_DELAY  = 3

MQTT_BROKER        = "10.1.193.80"
MQTT_PORT          = 1883
MQTT_TOPIC_BASE    = "classroom/room"

# ================= LOGGING =================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("camera_system")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "camera_detection.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ================= MQTT =================
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def publish_state(cam_id, payload):
    mqtt_client.publish(f"{MQTT_TOPIC_BASE}{cam_id}", payload, retain=True)
    logger.info(f"CAM={cam_id} | {payload}")

def publish_segment_state(cam_id, segment, state):
    payload = f"S{segment}_{'ON' if state else 'OFF'}"
    mqtt_client.publish(f"{MQTT_TOPIC_BASE}{cam_id}", payload, retain=True)
    logger.info(f"CAM={cam_id} | SEG={segment} | {'ON' if state else 'OFF'}")

# ================= YOLO (SEGMENTATION) =================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt")  # Replace with your trained segmentation model

last_detect_ts = {cid: 0 for cid in CAMERAS}
human_in_room  = {cid: False for cid in CAMERAS}

segment_state     = {3: {}}
segment_last_seen = {3: {}}

# ================= CPU MONITOR =================
def cpu_monitor(interval=5):
    process = psutil.Process(os.getpid())
    while True:
        sys_percent = psutil.cpu_percent()
        proc_percent = process.cpu_percent()
        logger.info(f"CPU | system={sys_percent:.1f}% | process={proc_percent:.1f}%")
        time.sleep(interval)

# ================= CAMERA THREAD =================
def camera_reader(cam_id, url, q):
    while True:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            logger.error(f"CAM_OPEN_FAIL | cam={cam_id}")
            time.sleep(2)
            continue

        logger.info(f"CAM_OPENED | cam={cam_id}")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"FRAME_FAIL | cam={cam_id}")
                cap.release()
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            if q.full():
                q.get()
            q.put(frame)

# ================= SEGMENTATION DRAW =================
def apply_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[:] = color
    frame[mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask]
    return frame

# ================= YOLO THREAD =================
def yolo_worker(q, cam_id, cam_logic):
    frame_count = 0
    logger.info(f"YOLO_START | cam={cam_id}")

    while True:
        if q.empty():
            time.sleep(0.01)
            continue

        frame = q.get()
        frame_count += 1

        if frame_count % (SKIP_FRAMES + 1):
            cv2.imshow(f"Camera {cam_id}", frame)
            cv2.waitKey(1)
            continue

        human_detected = False
        segments_detected = []

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        annotated = frame.copy()

        for r in results:
            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cam_logic.should_ignore(cx, cy):
                    continue

                human_detected = True

                # ---- Bounding Box ----
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, "Person",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )

                # ---- Segmentation Mask from YOLO ----
                if r.masks is not None:
                    mask = r.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    annotated = apply_mask(annotated, mask > 0.5)

                # ---- Segmentation Points from Camera Module ----
                if hasattr(cam_logic, "get_segments"):
                    segs = cam_logic.get_segments(cx, cy) or []
                else:
                    seg = cam_logic.get_segment(cx, cy)
                    segs = [seg] if seg else []

                for seg in segs:
                    if seg not in segments_detected:
                        segments_detected.append(seg)

        now = time.time()

        if human_detected:
            last_detect_ts[cam_id] = now

        if human_detected and not human_in_room[cam_id]:
            human_in_room[cam_id] = True
            publish_state(cam_id, "HUMAN_DETECTED")

        if not human_detected and human_in_room[cam_id]:
            if now - last_detect_ts[cam_id] > AUTO_OFF_SECONDS:
                human_in_room[cam_id] = False
                publish_state(cam_id, "NO_HUMAN")

        # ---- Segment state update for cam3 ----
        if cam_id == 3:
            prev = segment_state[cam_id]
            seen = segment_last_seen[cam_id]

            for seg in segments_detected:
                seen[seg] = now
                if not prev.get(seg):
                    prev[seg] = True
                    publish_segment_state(3, seg, True)

            for seg in list(prev.keys()):
                if seg not in segments_detected:
                    if now - seen.get(seg, 0) >= SEGMENT_OFF_DELAY:
                        if prev[seg]:
                            prev[seg] = False
                            publish_segment_state(3, seg, False)

        cv2.imshow(f"Camera {cam_id}", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            os._exit(0)

# ================= MAIN =================
def main():
    signal.signal(signal.SIGINT, lambda s, f: os._exit(0))

    threading.Thread(target=cpu_monitor, daemon=True).start()

    for cam_id, cfg in CAMERAS.items():
        try:
            cam_logic = importlib.import_module(f"cameras.{cfg['module']}")
        except ModuleNotFoundError:
            import types
            cam_logic = types.SimpleNamespace(
                should_ignore=lambda cx, cy: False,
                get_segment=lambda cx, cy: None
            )

        q = queue.Queue(maxsize=1)

        threading.Thread(
            target=camera_reader,
            args=(cam_id, cfg["url"], q),
            daemon=True
        ).start()

        threading.Thread(
            target=yolo_worker,
            args=(q, cam_id, cam_logic),
            daemon=True
        ).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()