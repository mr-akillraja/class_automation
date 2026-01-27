import re
import pandas as pd
from datetime import datetime

# ================= LOG FILE PATH =================
# Use raw string to avoid Windows backslash issues
LOG_FILE = r"C:\\Users\\Admin\\PyCharmMiscProject\\final_code\\logs\\camera_detection.log"

# ================= REGEX PATTERNS =================
# Updated to match actual log messages from your camera system
PATTERNS = {
    "cpu": re.compile(
        r"(?P<ts>.*?) \| INFO \| CPU \| system=(?P<sys>[\d.]+)% \| process=(?P<proc>[\d.]+)%"
    ),
    "room_on": re.compile(
        r"(?P<ts>.*?) \| INFO \| MQTT \| cam=(?P<cam>\d+) \| HUMAN_DETECTED"
    ),
    "room_off": re.compile(
        r"(?P<ts>.*?) \| INFO \| MQTT \| cam=(?P<cam>\d+) \| NO_HUMAN"
    ),
    "seg_on": re.compile(
        r"(?P<ts>.*?) \| INFO \| MQTT_SEG \| cam=(?P<cam>\d+) \| seg=(?P<seg>\d+) \| ON"
    ),
    "seg_off": re.compile(
        r"(?P<ts>.*?) \| INFO \| MQTT_SEG \| cam=(?P<cam>\d+) \| seg=(?P<seg>\d+) \| OFF"
    ),
}

# ================= HELPERS =================
def parse_time(ts):
    """Convert timestamp string to datetime object"""
    return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S,%f")

# ================= LOAD LOGS =================
cpu_rows = []
room_events = []
segment_events = []

with open(LOG_FILE, "r") as f:
    for line in f:
        for key, regex in PATTERNS.items():
            m = regex.search(line)
            if not m:
                continue

            ts = parse_time(m.group("ts"))

            if key == "cpu":
                cpu_rows.append({
                    "time": ts,
                    "system_cpu": float(m.group("sys")),
                    "process_cpu": float(m.group("proc")),
                })

            elif key in ("room_on", "room_off"):
                room_events.append({
                    "time": ts,
                    "cam": int(m.group("cam")),
                    "event": key,
                })

            elif key in ("seg_on", "seg_off"):
                segment_events.append({
                    "time": ts,
                    "cam": int(m.group("cam")),
                    "segment": int(m.group("seg")),
                    "event": key,
                })

# ================= CREATE DATAFRAMES =================
df_cpu = pd.DataFrame(cpu_rows)
df_rooms = pd.DataFrame(room_events)
df_segments = pd.DataFrame(segment_events)

# Sort only if the DataFrame is not empty
if not df_cpu.empty:
    df_cpu = df_cpu.sort_values("time")

if not df_rooms.empty:
    df_rooms = df_rooms.sort_values("time")

if not df_segments.empty:
    df_segments = df_segments.sort_values("time")

# ================= ROOM OCCUPANCY =================
occupancy = []

if not df_rooms.empty:
    for cam, grp in df_rooms.groupby("cam"):
        grp = grp.sort_values("time")
        start = None
        for _, row in grp.iterrows():
            if row["event"] == "room_on":
                start = row["time"]
            elif row["event"] == "room_off" and start:
                duration = (row["time"] - start).total_seconds()
                occupancy.append({
                    "cam": cam,
                    "start": start,
                    "end": row["time"],
                    "duration_sec": duration
                })
                start = None

df_occupancy = pd.DataFrame(occupancy)

# ================= SEGMENT USAGE =================
segment_usage = []

if not df_segments.empty:
    for (cam, seg), grp in df_segments.groupby(["cam", "segment"]):
        grp = grp.sort_values("time")
        start = None
        for _, row in grp.iterrows():
            if row["event"] == "seg_on":
                start = row["time"]
            elif row["event"] == "seg_off" and start:
                segment_usage.append({
                    "cam": cam,
                    "segment": seg,
                    "duration_sec": (row["time"] - start).total_seconds()
                })
                start = None

df_segment_usage = pd.DataFrame(segment_usage)

# ================= CPU SUMMARY =================
cpu_summary = df_cpu.describe() if not df_cpu.empty else pd.DataFrame()

# ================= OUTPUT =================
print("\n===== CPU SUMMARY =====")
if not cpu_summary.empty:
    print(cpu_summary[["system_cpu", "process_cpu"]])
else:
    print("No CPU data found.")

print("\n===== ROOM OCCUPANCY (seconds) =====")
if not df_occupancy.empty:
    print(df_occupancy.groupby("cam")["duration_sec"].sum())
else:
    print("No room occupancy events found.")

print("\n===== SEGMENT USAGE (seconds) =====")
if not df_segment_usage.empty:
    print(
        df_segment_usage
        .groupby(["cam", "segment"])["duration_sec"]
        .sum()
        .sort_values(ascending=False)
    )
else:
    print("No segment usage events found.")

# ================= OPTIONAL EXPORTS =================
df_cpu.to_csv("cpu_usage.csv", index=False)
df_occupancy.to_csv("room_occupancy.csv", index=False)
df_segment_usage.to_csv("segment_usage.csv", index=False)
