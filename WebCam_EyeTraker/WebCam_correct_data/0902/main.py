import mediapipe as mp
import cv2
import gaze
import csv
import time

# CSV 파일 초기화 및 헤더 작성
csv_filename = 'gaze_coordinates.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "gaze_x", "gaze_y", "left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"])


# gaze 데이터를 CSV에 저장하는 함수
def log_gaze_to_csv(timestamp, gaze_coords, left_pupil, right_pupil):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [timestamp, gaze_coords[0], gaze_coords[1], left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1]])


mp_face_mesh = mp.solutions.face_mesh  # Face Mesh 모델 초기화

# 카메라 스트림 열기
cap = cv2.VideoCapture(0)  # 카메라 인덱스 설정 (필요에 따라 1, 2, 3 등으로 변경)
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
            gaze_coords, left_pupil, right_pupil = gaze.gaze(image, results.multi_face_landmarks[0])  # 시선 추적

            if gaze_coords is not None:
                timestamp = time.time()  # 현재 타임스탬프 기록
                log_gaze_to_csv(timestamp, gaze_coords, left_pupil, right_pupil)  # 시선 및 눈동자 좌표를 CSV에 기록

                print(f"Gaze coordinates: {gaze_coords}")  # 시선 좌표 출력
                print(f"Left pupil coordinates: {left_pupil}")  # 왼쪽 눈동자 좌표 출력
                print(f"Right pupil coordinates: {right_pupil}\n")  # 오른쪽 눈동자 좌표 출력
            else:
                print("Gaze estimation failed for this frame.")

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
