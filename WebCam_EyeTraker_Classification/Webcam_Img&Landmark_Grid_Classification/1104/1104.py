import cv2
import csv
import mediapipe as mp
import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))

def gaze(frame, points):
    nose_tip = relative(points.landmark[4], frame.shape)
    chin = relative(points.landmark[152], frame.shape)
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)
    left_mouth_corner = relative(points.landmark[61], frame.shape)
    right_mouth_corner = relative(points.landmark[291], frame.shape)
    left_eye_outer = relative(points.landmark[33], frame.shape)
    right_eye_outer = relative(points.landmark[263], frame.shape)
    return left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer

# CSV 파일 설정
csv_filename = 'test_result/res.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y", "nose_tip_x", "nose_tip_y",
         "chin_x", "chin_y", "left_mouth_corner_x", "left_mouth_corner_y", "right_mouth_corner_x",
         "right_mouth_corner_y", "left_eye_outer_x", "left_eye_outer_y", "right_eye_outer_x", "right_eye_outer_y",
         "label"])

def log_gaze_to_csv(label, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner,
                    left_eye_outer, right_eye_outer):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1], nose_tip[0], nose_tip[1],
             chin[0], chin[1], left_mouth_corner[0], left_mouth_corner[1], right_mouth_corner[0],
             right_mouth_corner[1], left_eye_outer[0], left_eye_outer[1], right_eye_outer[0], right_eye_outer[1],
             label])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# 설정 값
max_sessions = 300
max_images_per_session = 45

# 캡처 위치 설정
width, height = 1960, 1080
window = np.zeros((height, width, 3), dtype=np.uint8)

start_x, start_y = 60, 40
horizontal_gap = 225
vertical_gap = 250
points = [(start_x + (i % 9) * horizontal_gap, start_y + (i // 9) * vertical_gap) for i in range(45)]

# 마우스 클릭 이벤트 핸들러
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = True

cv2.namedWindow('Webcam Window')
cv2.namedWindow('Points Window')

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    clicked = [False]
    cv2.setMouseCallback('Points Window', click_event, clicked)

    for session_count in range(1, max_sessions + 1):
        capture_count = 0

        # 세션 시작
        window.fill(0)
        cv2.putText(window, f'Session {session_count}', (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Points Window', window)

        # 클릭 대기
        while not clicked[0]:
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

        for idx, point in enumerate(points):
            window.fill(0)
            cv2.circle(window, point, 5, (0, 255, 0), -1)
            cv2.imshow('Points Window', window)

            while True:
                success, image = cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 웹캠 화면 표시
                cv2.imshow('Webcam Window', image)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer = gaze(
                        image, landmarks)

                    # 스페이스바를 눌러 캡처
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        frame_filename = f'test/img_{session_count}_{capture_count + 1:04d}.jpg'
                        log_gaze_to_csv(capture_count+1, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner,
                                        right_mouth_corner, left_eye_outer, right_eye_outer)
                        cv2.imwrite(frame_filename, image)
                        capture_count += 1
                        print(f"Captured image {capture_count} of session {session_count}: {frame_filename}")
                        break

                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()

            if capture_count >= max_images_per_session:
                break

        # 모든 점을 다 보여준 후 문구 출력
        print(f"Session {session_count} 완료")
        window.fill(0)
        cv2.putText(window, f"Session ({session_count}/{max_sessions})", (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Points Window', window)

        # 다음 세션으로 넘어가기 위한 클릭 대기
        clicked[0] = False
        while not clicked[0]:
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

cap.release()
cv2.destroyAllWindows()
