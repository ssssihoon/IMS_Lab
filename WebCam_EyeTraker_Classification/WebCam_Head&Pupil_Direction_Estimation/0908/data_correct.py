import cv2
import csv
import time
import mediapipe as mp

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)


def gaze(frame, points):
    '''
    Extracts and returns coordinates for specific facial landmarks.
    '''
    # 2D image points.
    nose_tip = relative(points.landmark[4], frame.shape)  # 코 끝 좌표
    chin = relative(points.landmark[152], frame.shape)  # 턱 끝 좌표

    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    left_mouth_corner = relative(points.landmark[61], frame.shape)  # 왼쪽 입 끝 좌표
    right_mouth_corner = relative(points.landmark[291], frame.shape)  # 오른쪽 입 끝 좌표

    left_eye_outer = relative(points.landmark[33], frame.shape)  # 왼쪽 눈 바깥쪽 좌표
    right_eye_outer = relative(points.landmark[263], frame.shape)  # 오른쪽 눈 바깥쪽 좌표

    # Draw circles at the pupil locations and the chin, nose tip, mouth corners, eye outer locations
    cv2.circle(frame, left_pupil, 3, (0, 255, 0), -1)  # Green dot for left pupil
    cv2.circle(frame, right_pupil, 3, (255, 0, 0), -1)  # Blue dot for right pupil
    cv2.circle(frame, nose_tip, 3, (0, 0, 255), -1)  # Red dot for nose tip
    cv2.circle(frame, chin, 3, (255, 255, 0), -1)  # Yellow dot for chin
    cv2.circle(frame, left_mouth_corner, 3, (0, 255, 255), -1)  # Cyan dot for left mouth corner
    cv2.circle(frame, right_mouth_corner, 3, (255, 0, 255), -1)  # Magenta dot for right mouth corner
    cv2.circle(frame, left_eye_outer, 3, (255, 255, 255), -1)  # White dot for left eye outer
    cv2.circle(frame, right_eye_outer, 3, (128, 128, 128), -1)  # Gray dot for right eye outer

    # Return the coordinates of interest
    return left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer


# CSV 파일 초기화 및 헤더 작성
csv_filename = 'up.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["timestamp", "left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y", "nose_tip_x", "nose_tip_y",
         "chin_x", "chin_y", "left_mouth_corner_x", "left_mouth_corner_y", "right_mouth_corner_x",
         "right_mouth_corner_y", "left_eye_outer_x", "left_eye_outer_y", "right_eye_outer_x", "right_eye_outer_y"])


# gaze 데이터를 CSV에 저장하는 함수
def log_gaze_to_csv(timestamp, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner,
                    left_eye_outer, right_eye_outer):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [timestamp, left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1], nose_tip[0], nose_tip[1], chin[0],
             chin[1], left_mouth_corner[0], left_mouth_corner[1], right_mouth_corner[0], right_mouth_corner[1],
             left_eye_outer[0], left_eye_outer[1], right_eye_outer[0], right_eye_outer[1]])


mp_face_mesh = mp.solutions.face_mesh

# 카메라 스트림 열기
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    frame_count = 0  # 프레임 카운터 초기화

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer = gaze(
                image, landmarks)

            if left_pupil is not None and right_pupil is not None:
                timestamp = time.time()
                log_gaze_to_csv(timestamp, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner,
                                right_mouth_corner, left_eye_outer, right_eye_outer)

                print(f"Left pupil coordinates: {left_pupil}")
                print(f"Right pupil coordinates: {right_pupil}")
                print(f"Nose tip coordinates: {nose_tip}")
                print(f"Chin coordinates: {chin}")
                print(f"Left mouth corner coordinates: {left_mouth_corner}")
                print(f"Right mouth corner coordinates: {right_mouth_corner}")
                print(f"Left eye outer coordinates: {left_eye_outer}")
                print(f"Right eye outer coordinates: {right_eye_outer}\n")

                # 프레임을 이미지 파일로 저장
                frame_filename = f'img_{frame_count:04d}.jpg'
                cv2.imwrite(frame_filename, image)
                frame_count += 1

        cv2.imshow('output window', image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()
