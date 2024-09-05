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

    # 2D pupil locations
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Draw circles at the pupil locations and the chin, nose tip locations
    cv2.circle(frame, left_pupil, 3, (0, 255, 0), -1)  # Green dot for left pupil
    cv2.circle(frame, right_pupil, 3, (255, 0, 0), -1)  # Blue dot for right pupil
    cv2.circle(frame, nose_tip, 3, (0, 0, 255), -1)  # Red dot for nose tip
    cv2.circle(frame, chin, 3, (255, 255, 0), -1)  # Yellow dot for chin

    # Return the coordinates of interest
    return left_pupil, right_pupil, nose_tip, chin

# CSV 파일 초기화 및 헤더 작성
csv_filename = 'up.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y", "nose_tip_x", "nose_tip_y", "chin_x", "chin_y"])

# gaze 데이터를 CSV에 저장하는 함수
def log_gaze_to_csv(timestamp, left_pupil, right_pupil, nose_tip, chin):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [timestamp, left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1], nose_tip[0], nose_tip[1], chin[0], chin[1]])

mp_face_mesh = mp.solutions.face_mesh  # Face Mesh 모델 초기화

# 카메라 스트림 열기
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 각 프레임에서 추적할 얼굴 수
        refine_landmarks=True,  # Face Mesh 모델에 홍채 랜드마크 포함
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # 프레임 입력이 없는 경우
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Face Mesh 모델을 위해 프레임을 RGB로 변환
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV를 위해 프레임을 다시 BGR로 변환

        if results.multi_face_landmarks:
            left_pupil, right_pupil, nose_tip, chin = gaze(image, results.multi_face_landmarks[0])  # 얼굴 랜드마크 추적

            # 스페이스바가 눌렸는지 체크
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # 스페이스바 키 코드
                timestamp = time.time()  # 현재 타임스탬프 기록
                log_gaze_to_csv(timestamp, left_pupil, right_pupil, nose_tip, chin)  # 좌표를 CSV에 기록

                print(f"Left pupil coordinates: {left_pupil}")  # 왼쪽 눈동자 좌표 출력
                print(f"Right pupil coordinates: {right_pupil}")  # 오른쪽 눈동자 좌표 출력
                print(f"Nose tip coordinates: {nose_tip}")  # 코 끝 좌표 출력
                print(f"Chin coordinates: {chin}\n")  # 턱 끝 좌표 출력

        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(image, (0, center_y), (width, center_y), (255, 0, 0), 2)  # 파란색 X축
        cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # 초록색 Y축

        # 화면에 이미지 표시
        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:  # Esc 키를 눌러 종료
            break

cap.release()
cv2.destroyAllWindows()
