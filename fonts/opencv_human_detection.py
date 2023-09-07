import cv2


# 마우스 이벤트 콜백 함수
def select_face(event, x, y, flags, param):
    global selected_face, start_x, start_y, end_x, end_y
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼을 누르면
        selected_face = None  # 선택된 얼굴 초기화
        start_x, start_y = x, y  # 드래그 시작 좌표 저장x
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼을 놓으면
        end_x, end_y = x, y  # 드래그 끝 좌표 저장
        selected_face = (start_x, start_y, end_x, end_y)  # 선택된 얼굴 좌표 저장


# 얼굴 저장 함수
def save_face(face, count):
    # 해당 이름의 폴더가 없다면 생성
    if not cv2.os.path.isdir(face_dirs + name):
        cv2.os.makedirs(face_dirs + name)

    # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
    file_name_path = face_dirs + name + "/user" + str(count) + ".jpg"
    cv2.imwrite(file_name_path, face)


# 얼굴만 저장하는 함수
def take_pictures(name):
    # 카메라 ON
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면
        if face_extractor(frame) is not None:
            count += 1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame), (200, 200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        if selected_face is not None:
            for x, y, w, h in faces:
                # 선택한 얼굴 영역 좌표
                selected_x1, selected_y1, selected_x2, selected_y2 = selected_face
                # 얼굴 영역이 선택한 영역과 겹치지 않으면 모자이크 처리
                if (
                    x + w < selected_x1
                    or x > selected_x2
                    or y + h < selected_y1
                    or y > selected_y2
                ):
                    face_mosaic = mosaic(face, (0, 0, w, h), 10)  # 모자이크 크기를 더 작게 조절
                    frame[y : y + h, x : x + w] = face_mosaic
                    save_face(face, count)

        cv2.putText(
            frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Face Cropper", frame)

        # 얼굴 사진 100장을 다 얻었거나 enter키 누르면 종료
        if cv2.waitKey(1) == 13 or count == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Colleting Samples Complete!!!")


# 카메라 영상을 받아올 객체 선언 및 설정(영상 소스, 해상도 설정)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# haar cascade 검출기 객체 선언
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")


# 모자이크 함수 정의
def mosaic(img, rect, size):
    x, y, w, h = rect
    small = cv2.resize(img[y : y + h, x : x + w], (size, size))  # 모자이크 크기 조절
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)


# 마우스 이벤트 콜백 함수 등록
cv2.namedWindow("original")
cv2.setMouseCallback("original", select_face)

selected_face = None
start_x, start_y, end_x, end_y = 0, 0, 0, 0

# 무한 루프
while True:
    ret, frame = capture.read()  # 카메라로부터 현재 영상을 받아 frame에 저장
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로 바꿔줌

    # scaleFactor를 높여 정확도를 상승시키고 minSize를 더 작게 조절
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    # 얼굴을 검출한 후 모자이크 처리
    for x, y, w, h in faces:
        if selected_face is not None:
            # 선택한 얼굴 영역 좌표
            selected_x1, selected_y1, selected_x2, selected_y2 = selected_face
            # 얼굴 영역이 선택한 영역과 겹치지 않으면 모자이크 처리
            if (
                x + w < selected_x1
                or x > selected_x2
                or y + h < selected_y1
                or y > selected_y2
            ):
                face = frame[y : y + h, x : x + w]
                face_mosaic = mosaic(face, (0, 0, w, h), 10)  # 모자이크 크기를 더 작게 조절
                frame[y : y + h, x : x + w] = face_mosaic

    cv2.imshow("original", frame)  # frame(카메라 영상)을 original 이라는 창에 띄워줌
    if cv2.waitKey(1) == ord("q"):  # 키보드의 q 를 누르면 무한 루프가 멈춤
        break

capture.release()  # 캡처 객체를 해제
cv2.destroyAllWindows()  # 모든 영상 창을 닫아줌
