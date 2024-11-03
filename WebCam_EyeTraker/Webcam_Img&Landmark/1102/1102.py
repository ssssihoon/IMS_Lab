import cv2
import csv
import mediapipe as mp

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
         "right_mouth_corner_y", "left_eye_outer_x", "left_eye_outer_y", "right_eye_outer_x", "right_eye_outer_y", "label"])

def log_gaze_to_csv(label, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner,
                    left_eye_outer, right_eye_outer):
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [left_pupil[0], left_pupil[1], right_pupil[0], right_pupil[1], nose_tip[0], nose_tip[1],
             chin[0], chin[1], left_mouth_corner[0], left_mouth_corner[1], right_mouth_corner[0],
             right_mouth_corner[1], left_eye_outer[0], left_eye_outer[1], right_eye_outer[0], right_eye_outer[1], label])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

session_count = 1
capture_count = 0


# <수정가능>
max_sessions = 50
max_images_per_session = 300


with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        if session_count > max_sessions:
            print("End sessions")
            break  # 세션이 끝나면 프로그램 종료

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
            left_pupil, right_pupil, nose_tip, chin, left_mouth_corner, right_mouth_corner, left_eye_outer, right_eye_outer = gaze(image, landmarks)

            # 스페이스바를 눌렀을 때만 이미지 저장
            if cv2.waitKey(1) & 0xFF == ord(' '):
                if capture_count < max_images_per_session:
                    frame_filename = f'test/img_{session_count}_{capture_count + 1:04d}.jpg'
                    log_gaze_to_csv(session_count, left_pupil, right_pupil, nose_tip, chin, left_mouth_corner,
                                    right_mouth_corner, left_eye_outer, right_eye_outer)

                    cv2.imwrite(frame_filename, image)
                    capture_count += 1
                    print(f"Captured image {capture_count} of session {session_count}: {frame_filename}")

                if capture_count == max_images_per_session:
                    session_count += 1
                    capture_count = 0
                    print(f"Session {session_count - 1} completed. Moving to session {session_count}.")

        cv2.imshow('output window', image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()
