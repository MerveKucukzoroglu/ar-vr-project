import cv2
import mediapipe as mp
import math

class FaceLandmarkDetector:
    def __init__(self, static_mode=False, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )
                self.get_face_shape(landmarks, frame)
        return frame

    def get_distance(self, p1, p2, w, h):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.hypot(x2 - x1, y2 - y1)

    def get_face_shape(self, landmarks, frame):
        h, w, _ = frame.shape

        top = landmarks.landmark[10]
        chin = landmarks.landmark[152]
        left = landmarks.landmark[234]
        right = landmarks.landmark[454]
        left_cheek = landmarks.landmark[93]
        right_cheek = landmarks.landmark[323]

        face_height = self.get_distance(top, chin, w, h)
        face_width = self.get_distance(left, right, w, h)
        cheek_width = self.get_distance(left_cheek, right_cheek, w, h)

        ratio = face_width / face_height

        # Basic classification based on width/height ratio
        if ratio > 0.9:
            shape = "Round"
        elif ratio > 0.8:
            shape = "Square"
        elif ratio > 0.7:
            shape = "Oval"
        elif ratio <= 0.6:
            shape = "Long"
        else:
            shape = "Unknown"

        cv2.putText(frame, f"Face Shape: {shape}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return shape
