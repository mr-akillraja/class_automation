import cv2
import time
import threading
import queue
import torch
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import numpy as np

# ================== MQTT CONFIG ==================
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC_BASE = "classroom/room"

# ================== CAMERA CONFIG =================
CAMERA_URLS = [
    "rtsp://admin:Akill%40123@192.168.1.126:554/unicaststream/1",
]

CONFIDENCE = 0.25
IMG_SIZE = 320
FRAME_WIDTH = 480
FRAME_HEIGHT = 270
SKIP_FRAMES = 2

AUTO_OFF_SECONDS = 1
MIN_STATE_INTERVAL = 0.5

# ================== ZONES ==================
ZONE_1 = np.array([(3, 83), (249, 36), (479, 153), (475, 264), (5, 268)], np.int32)
ZONE_2 = np.array([(250, 33), (366, 20), (475, 62), (479, 152), (247, 34)], np.int32)

# ================== STATE ==================
last_state = {"zone1": False, "zone2": False}
last_detect_ts = {"zone1": 0, "zone2": 0}
last_change_ts = {"zone1": 0, "zone2": 0}

# ================== MQTT ==================
mqtt_client = mqtt.Client(client_id=f"YOLO_PC_{int(time.time())}")
mqtt_connected = False

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        print("[MQTT] Connected", flush=True)
    else:
        print(f"[MQTT] Failed rc={rc}", flush=True)

mqtt_client.on_connect = on_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT ERROR] {e}")
    print("[MQTT] Running offline mode")

def publish_state(zone, state):
    now = time.time()

    if now - last_change_ts[zone] < MIN_STATE_INTERVAL:
        return

    topic = f"{MQTT_TOPIC_BASE}/{zone}"
    payload = "1" if state else "0"

    if mqtt_connected:
        mqtt_client.publish(topic, payload, retain=True)
    else:
        print(f"[MQTT OFFLINE] {topic} -> {payload}")

    last_change_ts[zone] = now
    last_state[zone] = state

    print(f"[STATE] {zone} -> {payload}")

# ================== YOLO ==================
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("[YOLO] Device:", DEVICE)

model = YOLO("yolo11x.pt")

# ================== CAMERA THREAD ==================
def camera_reader(url, frame_queue):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[CAM] Stream started")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[CAM] Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_queue.full():
            frame_queue.get_nowait()

        frame_queue.put(frame)

# ================== YOLO THREAD ==================
def yolo_worker(frame_queue):
    print("[YOLO] Worker started")

    frame_count = 0
    cv2.namedWindow("CAM", cv2.WINDOW_NORMAL)

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        display_frame = frame.copy()
        frame_count += 1

        # Draw zones
        cv2.polylines(display_frame, [ZONE_1], True, (255, 0, 0), 2)
        cv2.polylines(display_frame, [ZONE_2], True, (0, 0, 255), 2)

        # Skip frames
        if frame_count % (SKIP_FRAMES + 1) != 0:
            cv2.imshow("CAM", display_frame)
            cv2.waitKey(1)
            continue

        z1_detected = False
        z2_detected = False

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])

                    if cls != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Draw box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Zone checks
                    if cv2.pointPolygonTest(ZONE_1, (cx, cy), False) >= 0:
                        z1_detected = True

                    if cv2.pointPolygonTest(ZONE_2, (cx, cy), False) >= 0:
                        z2_detected = True

        now = time.time()

        # ===== ZONE 1 =====
        if z1_detected:
            last_detect_ts["zone1"] = now
            if not last_state["zone1"]:
                print("[ZONE1] HUMAN DETECTED")
                publish_state("zone1", True)

        if last_state["zone1"] and (now - last_detect_ts["zone1"] > AUTO_OFF_SECONDS):
            print("[ZONE1] HUMAN LEFT")
            publish_state("zone1", False)

        # ===== ZONE 2 =====
        if z2_detected:
            last_detect_ts["zone2"] = now
            if not last_state["zone2"]:
                print("[ZONE2] HUMAN DETECTED")
                publish_state("zone2", True)

        if last_state["zone2"] and (now - last_detect_ts["zone2"] > AUTO_OFF_SECONDS):
            print("[ZONE2] HUMAN LEFT")
            publish_state("zone2", False)

        cv2.imshow("CAM", display_frame)
        cv2.waitKey(1)

# ================== MAIN ==================
def main():
    print("[SYSTEM] Starting YOLO → MQTT")

    q = queue.Queue(maxsize=1)

    threading.Thread(target=camera_reader, args=(CAMERA_URLS[0], q), daemon=True).start()
    threading.Thread(target=yolo_worker, args=(q,), daemon=True).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()