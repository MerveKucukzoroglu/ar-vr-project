import cv2
import numpy as np

def get_skin_undertone(face_region):
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_face[:, :, 0])

    if 0 <= avg_hue <= 10 or 170 <= avg_hue <= 180:
        return 'Cool'
    elif 11 <= avg_hue <= 50:
        return 'Warm'
    elif 51 <= avg_hue <= 120:
        return 'Olive'
    else:
        return 'Neutral'
