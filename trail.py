
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

# ================== LOW LATENCY FIX ==================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "reorder_queue_size;0|"
    "threads;1"
)

# ================== PATH FIX ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ================= CONFIG =================
FRAME_WIDTH = 480
FRAME_HEIGHT = 270
CONFIDENCE = 0.25
IMG_SIZE = 640
SKIP_FRAMES = 2
AUTO_OFF_SECONDS = 6
SEGMENT_OFF_DELAY = 3

MQTT_BROKER = "10.1.193.56"
MQTT_PORT = 1883
MQTT_TOPIC_BASE = "classroom/room"

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

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ================= MQTT =================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def publish_state(cam_id, payload):
    mqtt_client.publish(f"{MQTT_TOPIC_BASE}{cam_id}", payload, retain=True)
    logger.info(f"CAM={cam_id} | {payload}")

def publish_segment_state(cam_id, segment, state):
    payload = f"S{segment}_{'ON' if state else 'OFF'}"
    mqtt_client.publish(f"{MQTT_TOPIC_BASE}{cam_id}", payload, retain=True)
    logger.info(f"CAM={cam_id} | SEG={segment} | {payload}")

# ================= YOLO =================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt")
model.to(DEVICE)

# ================= ONLY CAM 3 =================
CAM_ID = 3
CAM_CFG = CAMERAS[CAM_ID]

last_detect_ts = {CAM_ID: 0}
human_in_room = {CAM_ID: False}
segment_state = {CAM_ID: {}}
segment_last_seen = {CAM_ID: {}}

# ================= CPU MONITOR =================
def cpu_monitor(interval=5):
    process = psutil.Process(os.getpid())
    process.cpu_percent()
    while True:
        logger.info(
            f"CPU | system={psutil.cpu_percent():.1f}% | "
            f"process={process.cpu_percent():.1f}%"
        )
        time.sleep(interval)

# ================= CAMERA THREAD =================
def camera_reader(cam_id, cfg, q):
    if "sub_url" not in cfg:
        url = cfg["url"]
        cfg["sub_url"] = url.replace("/102", "/101") if "/102" in url else url

    stream_url = cfg["sub_url"]

    while True:
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

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
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass

            q.put_nowait(frame)

# ================= YOLO THREAD =================
def yolo_worker(q, cam_id, cam_logic):
    frame_count = 0
    logger.info(f"YOLO_START | cam={cam_id}")

    while True:
        try:
            frame = q.get(timeout=1)
        except queue.Empty:
            continue

        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1):
            continue

        human_detected = False
        detected_segments = []

        with torch.no_grad():
            results = model.predict(
                frame,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cam_logic.should_ignore(cx, cy):
                    continue

                human_detected = True

                if hasattr(cam_logic, "get_segments"):
                    segs = cam_logic.get_segments(cx, cy) or []
                else:
                    seg = cam_logic.get_segment(cx, cy)
                    segs = [(cam_id, seg)] if seg else []

                for entry in segs:
                    if entry not in detected_segments:
                        detected_segments.append(entry)

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

        for tgt_cam, seg in detected_segments:
            segment_last_seen[tgt_cam][seg] = now
            if not segment_state[tgt_cam].get(seg):
                segment_state[tgt_cam][seg] = True
                publish_segment_state(tgt_cam, seg, True)

        for seg in list(segment_state[cam_id]):
            if (cam_id, seg) not in detected_segments:
                if now - segment_last_seen[cam_id].get(seg, 0) >= SEGMENT_OFF_DELAY:
                    if segment_state[cam_id][seg]:
                        segment_state[cam_id][seg] = False
                        publish_segment_state(cam_id, seg, False)

# ================= MAIN =================
def main():
    signal.signal(signal.SIGINT, lambda s, f: os._exit(0))
    threading.Thread(target=cpu_monitor, daemon=True).start()

    try:
        cam_logic = importlib.import_module(f"cameras.{CAM_CFG['module']}")
    except Exception:
        import types
        cam_logic = types.SimpleNamespace(
            should_ignore=lambda cx, cy: False,
            get_segment=lambda cx, cy: None
        )

    q = queue.Queue(maxsize=1)

    threading.Thread(
        target=camera_reader,
        args=(CAM_ID, CAM_CFG, q),
        daemon=True
    ).start()

    threading.Thread(
        target=yolo_worker,
        args=(q, CAM_ID, cam_logic),
        daemon=True
    ).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

