
import cv2

# ================= IGNORE POLYGON =================
# Region where human detection should be ignored for Camera 1
IGNORE_POLYGON = [
   (127, 3), 
   (127, 74), 
   (167, 64), 
   (168, 1)
]

# ================== SEGMENTS =================
# For Cam1, we may not have multiple segments, return None
POLYGONS = []  # no additional segments, only ignore zone

# Check if a point should be ignored
def should_ignore(cx, cy):
    return point_in_polygon((cx, cy), IGNORE_POLYGON)

# No segments defined, return None
def get_segment(cx, cy):
    return None

# Draw polygon overlay (only ignore region)
def draw_overlay(frame, detected_segment=None):
    # Draw ignore polygon
    for i in range(len(IGNORE_POLYGON)):
        cv2.line(frame, IGNORE_POLYGON[i], IGNORE_POLYGON[(i+1)%len(IGNORE_POLYGON)], (0,255,255), 2)

    # Optionally, label polygon
    cv2.putText(frame, "IGNORE ZONE", IGNORE_POLYGON[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("CAM 1 - Ignore Zone", frame)
    cv2.waitKey(1)

# Ray-casting algorithm to check if a point is inside a polygon
def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    px, py = poly[0]
    for i in range(1, len(poly) + 1):
        sx, sy = poly[i % len(poly)]
        if ((sy > y) != (py > y)) and (x < (px - sx) * (y - sy) / (py - sy + 1e-6) + sx):
            inside = not inside
        px, py = sx, sy
    return inside
