
import math
import cv2

def get_distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

def get_angle(a, b, c, w, h):
    ax, ay = int(a.x * w), int(a.y * h)
    bx, by = int(b.x * w), int(b.y * h)
    cx, cy = int(c.x * w), int(c.y * h)

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot_product = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)

    if mag_ab * mag_cb == 0:
        return 0
    cos_angle = dot_product / (mag_ab * mag_cb)
    angle = math.acos(min(1, max(-1, cos_angle)))
    return math.degrees(angle)

def get_face_shape(landmarks, w, h):
    chin = landmarks.landmark[152]
    forehead = landmarks.landmark[10]
    left_cheek = landmarks.landmark[234]
    right_cheek = landmarks.landmark[454]
    left_temple = landmarks.landmark[127]
    right_temple = landmarks.landmark[356]
    left_jaw = landmarks.landmark[172]
    right_jaw = landmarks.landmark[400]

    # Measurements
    face_length = get_distance(forehead, chin, w, h)
    cheekbone_width = get_distance(left_cheek, right_cheek, w, h)
    jaw_width = get_distance(left_jaw, right_jaw, w, h)
    forehead_width = get_distance(left_temple, right_temple, w, h)

    # Ratios
    width_to_length_ratio = cheekbone_width / face_length
    jaw_to_cheek_ratio = jaw_width / cheekbone_width
    forehead_to_jaw_ratio = forehead_width / jaw_width

    # Angle for jawline
    jaw_angle_left = get_angle(left_cheek, left_jaw, chin, w, h)
    jaw_angle_right = get_angle(right_cheek, right_jaw, chin, w, h)
    avg_jaw_angle = (jaw_angle_left + jaw_angle_right) / 2

    # Heuristic-based classification
    if abs(face_length - cheekbone_width) < 20 and avg_jaw_angle > 155:
        return "Round"
    elif abs(face_length - cheekbone_width) < 20 and avg_jaw_angle <= 155:
        return "Square"
    elif forehead_width > cheekbone_width and cheekbone_width > jaw_width and forehead_to_jaw_ratio > 1.2:
        return "Heart"
    elif cheekbone_width > forehead_width and cheekbone_width > jaw_width and forehead_to_jaw_ratio > 1.1:
        return "Diamond"
    elif width_to_length_ratio < 0.6:
        return "Oblong"
    elif jaw_width > cheekbone_width and jaw_to_cheek_ratio > 1.05:
        return "Triangle"
    elif forehead_width > jaw_width and forehead_to_jaw_ratio > 1.3:
        return "Inverted Triangle"
    else:
        return "Oval"

import cv2

def draw_face_shape_overlay(frame, landmarks, w, h):
    def to_px(pt):
        return int(pt.x * w), int(pt.y * h)

    # Key landmarks
    chin = landmarks.landmark[152]
    forehead = landmarks.landmark[10]
    left_cheek = landmarks.landmark[234]         # left cheekbone
    right_cheek = landmarks.landmark[454]        # right cheekbone
    left_temple = landmarks.landmark[127]
    right_temple = landmarks.landmark[356]
    jaw_corner_left = landmarks.landmark[172]    # left jaw corner
    jaw_corner_right = landmarks.landmark[397]   # right jaw corner

    # Convert to pixel coordinates
    pt_chin = to_px(chin)
    pt_forehead = to_px(forehead)
    pt_left_cheek = to_px(left_cheek)
    pt_right_cheek = to_px(right_cheek)
    pt_left_temple = to_px(left_temple)
    pt_right_temple = to_px(right_temple)
    pt_jaw_corner_left = to_px(jaw_corner_left)
    pt_jaw_corner_right = to_px(jaw_corner_right)

    # Face measurement lines
    cv2.line(frame, pt_forehead, pt_chin, (255, 100, 100), 2)                # vertical face height
    cv2.line(frame, pt_left_cheek, pt_right_cheek, (100, 255, 100), 2)       # cheekbone width
    cv2.line(frame, pt_jaw_corner_left, pt_jaw_corner_right, (100, 100, 255), 2)  # jaw width
    cv2.line(frame, pt_left_temple, pt_right_temple, (255, 255, 0), 2)       # forehead width

    # Jawline triangles
    cv2.line(frame, pt_left_cheek, pt_jaw_corner_left, (200, 200, 255), 1)
    cv2.line(frame, pt_jaw_corner_left, pt_chin, (200, 200, 255), 1)
    cv2.line(frame, pt_right_cheek, pt_jaw_corner_right, (200, 200, 255), 1)
    cv2.line(frame, pt_jaw_corner_right, pt_chin, (200, 200, 255), 1)

    # Debugging dots (optional - you can remove if not needed)
    cv2.circle(frame, pt_left_cheek, 3, (0, 255, 0), -1)         # green
    cv2.circle(frame, pt_right_cheek, 3, (0, 255, 0), -1)        # green
    cv2.circle(frame, pt_jaw_corner_left, 3, (255, 0, 0), -1)    # blue
    cv2.circle(frame, pt_jaw_corner_right, 3, (255, 0, 0), -1)   # blue
    cv2.circle(frame, pt_chin, 3, (200, 200, 200), -1)           # gray
    cv2.circle(frame, pt_forehead, 3, (200, 200, 200), -1)       # gray
