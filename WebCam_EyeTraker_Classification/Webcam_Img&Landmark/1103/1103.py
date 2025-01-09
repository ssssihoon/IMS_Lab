import cv2
import numpy as np

width, height = 1960, 1080
window = np.zeros((height, width, 3), dtype=np.uint8)

start_x, start_y = 60, 40
horizontal_gap = 225
vertical_gap = 250

points = [(start_x + (i % 9) * horizontal_gap, start_y + (i // 9) * vertical_gap) for i in range(45)]


# 마우스 클릭 이벤트 핸들러
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        param[0] = True  # 클릭 상태를 True로 설정


for section in range(300):  # 3번 반복
    clicked = [False]  # 클릭 상태를 저장할 리스트

    # 섹션 번호 표시
    window.fill(0)
    cv2.putText(window, f'Section {section + 1}', (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Points Window', window)

    cv2.setMouseCallback('Points Window', click_event, clicked)  # 마우스 콜백 설정

    # 마우스 클릭 대기
    while not clicked[0]:
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            exit()

    for point in points:
        window.fill(0)
        cv2.circle(window, point, 5, (0, 255, 0), -1)

        cv2.imshow('Points Window', window)

        key = cv2.waitKey(0)
        if key != 13:
            cv2.destroyAllWindows()
            exit()

    window.fill(0)  # 화면 지우기
    cv2.putText(window, f'Section {section + 1}', (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Points Window', window)

    # 마우스 클릭 대기
    clicked[0] = False  # 클릭 상태 초기화
    while not clicked[0]:
        key = cv2.waitKey(1)
        if key == 27:  # ESC 키로 종료
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
