# H523


import cv2

# ================== POLYGONS SEGMENTS 1-4 =================
POLYGONS = [
[(1, 126), (266, 40), (475, 138), (476, 262), (4, 134)],
[(267, 37), (322, 32), (472, 70), (477, 135), (268, 38)],
[(2, 105), (218, 48), (478, 156), (476, 265), (6, 269)],
[(222, 48), (372, 37), (477, 73), (479, 152), (219, 53)]
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
