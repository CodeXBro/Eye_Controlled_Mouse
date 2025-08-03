import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Iris landmarks (Right eye: 474-478)
        for idx, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            if idx == 1:  # Tracking one specific point
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        # Detect blink using left eye landmarks (145 and 159)
        left_eye = [landmarks[145], landmarks[159]]
        for eye_point in left_eye:
            x = int(eye_point.x * frame_w)
            y = int(eye_point.y * frame_h)
            cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

        # Simple blink detection
        if abs(left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
