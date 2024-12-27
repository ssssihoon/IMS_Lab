import cv2
import csv
import mediapipe as mp
import numpy as np
import random

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
        ["session", "point", "frame", "left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y", "nose_tip_x", "nose_tip_y",
         "chin_x", "chin_y", "left_mouth_corner_x", "left_mouth_corner_y", "right_mouth_corner_x",
         "right_mouth_corner_y", "left_eye_outer_x", "left_eye_outer_y", "right_eye_outer_x", "right_eye_outer_y"])

def log_gaze_to_csv(session, point, frame_num, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner,
                    left_eye_outer, right_eye_outer):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [session, point, frame_num, left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1], nose_tip[0], nose_tip[1],
             chin[0], chin[1], left_mouth_corner[0], left_mouth_corner[1], right_mouth_corner[0],
             right_mouth_corner[1], left_eye_outer[0], left_eye_outer[1], right_eye_outer[0], right_eye_outer[1]])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# 설정 값
max_sessions = 50
max_points_per_session = 45
window_width, window_height = 1960, 1080
window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

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
        # 세션 시작
        window.fill(0)
        cv2.putText(window, f'Session {session_count}', (window_width // 2 - 100, window_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Points Window', window)

        # 클릭 대기
        while not clicked[0]:
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

        # 랜덤으로 점 위치 생성
        points = [(random.randint(50, window_width - 50), random.randint(50, window_height - 50)) for _ in range(max_points_per_session)]

        for point_idx, point in enumerate(points, 1):
            window.fill(0)
            cv2.circle(window, point, 5, (0, 255, 0), -1)
            cv2.imshow('Points Window', window)

            # 이미지 캡처 및 저장
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

                    # 필요한 랜드마크를 추출
                    try:
                        left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer = gaze(
                            image, landmarks
                        )

                        # 주요 랜드마크의 좌표가 모두 유효한 경우에만 촬영
                        if all(coord is not None for coord in [left_pupil, right_pupil, nose_tip, chin,
                                                               left_mouth_corner, right_mouth_corner,
                                                               left_eye_outer, right_eye_outer]):
                            if cv2.waitKey(1) & 0xFF == ord(' '):
                                frame_filename = f'test/img_{session_count}_({point_idx}).jpg'
                                log_gaze_to_csv(session_count, point_idx, 1, left_pupil, right_pupil, nose_tip, chin,
                                                left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer)
                                cv2.imwrite(frame_filename, image)
                                print(f"Captured point {point_idx} in session {session_count}")
                                break
                    except KeyError:
                        # 특정 랜드마크가 누락된 경우 패스
                        print("Some landmarks are missing, skipping capture.")

                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    exit()

        # 모든 점을 다 보여준 후 문구 출력
        print(f"Session {session_count} 완료")
        window.fill(0)
        cv2.putText(window, f"Session ({session_count}/{max_sessions}) finish", (window_width // 2 - 100, window_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Points Window', window)

        # 다음 세션으로 넘어가기 위한 클릭 대기
        clicked[0] = False
        while not clicked[0]:
            if cv2.waitKey(1) & 0xFF == 27:
                exit()

cap.release()
cv2.destroyAllWindows()
