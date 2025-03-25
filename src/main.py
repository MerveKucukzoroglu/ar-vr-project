# src/main.py

import cv2
from face_detection.detector import FaceLandmarkDetector

def run():
    detector = FaceLandmarkDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera read failed.")
            break

        frame = detector.detect(frame)
        cv2.imshow('AI/AR/VR Project - Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
