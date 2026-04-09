#H516
import cv2

IGNORE_POLYGON = [
   (419, 29), 
   (413, 91), 
   (451, 101),
     (461, 42)
]

def should_ignore(cx, cy):
    return point_in_polygon((cx, cy), IGNORE_POLYGON)

def get_segment(cx, cy):
    return None

def draw_overlay(frame, detected_segment=None):
    for i in range(len(IGNORE_POLYGON)):
        cv2.line(frame,
                 IGNORE_POLYGON[i],
                 IGNORE_POLYGON[(i + 1) % len(IGNORE_POLYGON)],
                 (255, 0, 0), 2)

    cv2.putText(frame, "IGNORE ZONE",
                IGNORE_POLYGON[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("CAM 2", frame)
    cv2.waitKey(1)

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
