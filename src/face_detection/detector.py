
import cv2
import mediapipe as mp
from .face_shape import get_face_shape, draw_face_shape_overlay
from .skin_tone import get_skin_undertone

class FaceLandmarkDetector:
    def __init__(self, static_mode=False, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
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
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                               self.drawing_spec, self.drawing_spec)

                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = self.get_face_bounding_box(face_landmarks, w, h)

                if y_max > y_min and x_max > x_min:
                    face_region = frame[y_min:y_max, x_min:x_max]
                    undertone = get_skin_undertone(face_region)
                    face_shape = get_face_shape(face_landmarks, w, h)

                    cv2.putText(frame, f'Skin Tone: {undertone}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    cv2.putText(frame, f'Face Shape: {face_shape}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    draw_face_shape_overlay(frame, face_landmarks, w, h)

        return frame

    def get_face_bounding_box(self, face_landmarks, w, h):
        x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
        y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
        x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
        y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
        return x_min, y_min, x_max, y_max
