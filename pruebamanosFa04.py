
"""""
Based on the works of:
https://www.youtube.com/@OMES-va


"""""

import cv2
import mediapipe as mp

# Initialize the MediaPipe Face Mesh and Hands models and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Ensure the FACE_CONNECTIONS attribute is available for Face Mesh
if hasattr(mp_face_mesh, 'FACEMESH_TESSELATION'):
    face_connections = mp_face_mesh.FACEMESH_TESSELATION
elif hasattr(mp_face_mesh, 'FACE_CONNECTIONS'):
    face_connections = mp_face_mesh.FACE_CONNECTIONS
else:
    raise AttributeError("The required FACE_CONNECTIONS attribute is missing in MediaPipe Face Mesh module.")

# Ensure the HAND_CONNECTIONS attribute is available for Hands
if hasattr(mp_hands, 'HAND_CONNECTIONS'):
    hand_connections = mp_hands.HAND_CONNECTIONS
else:
    raise AttributeError("The required HAND_CONNECTIONS attribute is missing in MediaPipe Hands module.")

# Start capturing video from the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set up the window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 400, 400)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh, \
    mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       # Process face mesh
        results_face = face_mesh.process(frame_rgb)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Dibujar landmarks faciales con color semitransparente
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), radius=1, color=(0, 255, 255), thickness=-1)  # Amarillo con transparencia


        # Process hands
        results_hands = hands.process(frame_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    hand_connections,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

        # Resize the frame to 400x400 pixels
        frame_resized = cv2.resize(frame, (400, 400))

        # Display the resized frame
        cv2.imshow("Frame", frame_resized)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press Esc to exit
            break

cap.release()
cv2.destroyAllWindows()
