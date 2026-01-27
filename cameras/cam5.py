# H523


import cv2

# ================== POLYGONS SEGMENTS 1-4 =================
POLYGONS = [
 [(2, 106), (111, 62), (227, 89), (2, 178), (1, 106)],
 [(2, 180), (4, 267), (477, 267), (479, 151), (233, 92)],
 [(113, 59), (191, 34), (278, 56), (230, 86), (120, 59)],
 [(235, 86), (277, 49), (479, 89), (475, 148), (239, 89)],
]

# ================== IGNORE ZONE (DEFAULT NONE) =================
def should_ignore(cx, cy):
    return False

# ================== GET SEGMENT =================
def get_segment(cx, cy):
    for idx, poly in enumerate(POLYGONS, start=1):
        if point_in_polygon((cx, cy), poly):
            return idx
    return None

# ================== DRAW POLYGON OVERLAY =================
def draw_overlay(frame, detected_segment=None):
    for idx, poly in enumerate(POLYGONS, start=1):
        # Draw polygon
        for i in range(len(poly)):
            cv2.line(frame, poly[i], poly[(i+1)%len(poly)], (0,255,255), 2)
        # Label each polygon with its segment number
        cv2.putText(frame, f"SEG {idx}", poly[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Highlight detected segment if any
    if detected_segment:
        cv2.putText(frame, f"DETECTED SEG {detected_segment}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

    cv2.imshow("CAM 3", frame)
    cv2.waitKey(1)

# ================== POINT IN POLYGON =================
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
