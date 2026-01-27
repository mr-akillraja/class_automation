import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


LOG_FILE =  r"C:\\Users\\Admin\\PyCharmMiscProject\\final_code\\logs\\camera_detection.log"


sns.set_style("whitegrid")


# ================= REGEX =================
PATTERNS = {
    "cpu": re.compile(
        r"(?P<ts>.*?) \| INFO \| CPU \| system=(?P<sys>[\d.]+)% \| process=(?P<proc>[\d.]+)%"
    ),
    "room_on": re.compile(
        r"(?P<ts>.*?) \| INFO \| ROOM_ON\s+\| cam=(?P<cam>\d+)"
    ),
    "room_off": re.compile(
        r"(?P<ts>.*?) \| INFO \| ROOM_OFF\s+\| cam=(?P<cam>\d+)"
    ),
    "seg_on": re.compile(
        r"(?P<ts>.*?) \| INFO \| SEG_ON\s+\| cam=(?P<cam>\d+) \| seg=(?P<seg>\d+)"
    ),
    "seg_off": re.compile(
        r"(?P<ts>.*?) \| INFO \| SEG_OFF\s+\| cam=(?P<cam>\d+) \| seg=(?P<seg>\d+)"
    ),
}


# ================= HELPERS =================
def parse_time(ts):
    return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S,%f")


# ================= PARSE LOG =================
cpu, room, seg = [], [], []

with open(LOG_FILE) as f:
    for line in f:
        for key, rx in PATTERNS.items():
            m = rx.search(line)
            if not m:
                continue

            t = parse_time(m.group("ts"))

            if key == "cpu":
                cpu.append({
                    "time": t,
                    "System CPU %": float(m.group("sys")),
                    "Process CPU %": float(m.group("proc")),
                })

            elif key in ("room_on", "room_off"):
                room.append({
                    "time": t,
                    "cam": int(m.group("cam")),
                    "event": key,
                })

            elif key in ("seg_on", "seg_off"):
                seg.append({
                    "time": t,
                    "cam": int(m.group("cam")),
                    "segment": int(m.group("seg")),
                    "event": key,
                })


# ================= CREATE DATAFRAMES SAFELY =================
df_cpu = pd.DataFrame(cpu, columns=["time", "System CPU %", "Process CPU %"]).sort_values("time") if cpu else pd.DataFrame(columns=["time", "System CPU %", "Process CPU %"])
df_room = pd.DataFrame(room, columns=["time", "cam", "event"]).sort_values("time") if room else pd.DataFrame(columns=["time", "cam", "event"])
df_seg  = pd.DataFrame(seg, columns=["time", "cam", "segment", "event"]).sort_values("time") if seg else pd.DataFrame(columns=["time", "cam", "segment", "event"])


# ================= PLOTS =================

# 1️⃣ CPU Usage Timeline (Line Plot)
if not df_cpu.empty:
    plt.figure(figsize=(12, 5))
    plt.plot(df_cpu["time"], df_cpu["System CPU %"], label="System CPU", color='blue')
    plt.plot(df_cpu["time"], df_cpu["Process CPU %"], label="Process CPU", color='green')
    plt.title("CPU Utilization Over Time")
    plt.xlabel("Time")
    plt.ylabel("CPU %")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2️⃣ Camera Detection vs CPU Utilization
if not df_cpu.empty and not df_room.empty:
    plt.figure(figsize=(12, 5))
    
    # Plot CPU utilization
    plt.plot(df_cpu["time"], df_cpu["System CPU %"], label="System CPU", color='blue')
    plt.plot(df_cpu["time"], df_cpu["Process CPU %"], label="Process CPU", color='green')
    
    # Overlay camera detection events
    for cam in df_room['cam'].unique():
        cam_events = df_room[df_room['cam'] == cam]
        # Plot room_on events as markers
        plt.scatter(
            cam_events[cam_events['event'] == 'room_on']['time'], 
            [0]*len(cam_events[cam_events['event'] == 'room_on']),  # Plot at y=0 for visibility
            label=f'Camera {cam} ON',
            marker='^',
            s=50
        )
        # Plot room_off events as markers
        plt.scatter(
            cam_events[cam_events['event'] == 'room_off']['time'], 
            [0]*len(cam_events[cam_events['event'] == 'room_off']),
            label=f'Camera {cam} OFF',
            marker='v',
            s=50
        )
    
    plt.title("CPU Utilization and Camera Detection Events Over Time")
    plt.xlabel("Time")
    plt.ylabel("CPU %")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()
