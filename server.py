import socket
import cv2
import pickle
import struct
import imutils
import cv2
import dlib
import time
import datetime
from imutils import face_utils
from scipy.spatial import distance as dist


MINIMUM_EAR = 310
MAXIMUM_FRAME_COUNT = 30
MAXIMUM_UNRECOGNIZED_COUNT = 50
EYE_CLOSED_COUNTER = 0
UNRECOGNIZED_COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_2.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


ip = '0.0.0.0'  # ip 주소  # 스프링서버 115.85.182.194
port = 8090  # port 번호  # 스프링서버 포트 8080
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, port))
s.listen(10)
print('클라이언트 연결 대기')

conn, addr = s.accept()
print(addr)

data = b""
payload_size = struct.calcsize(">L")

print('Connected by', addr)

# javaip = '115.85.182.194' # ip 주소
# javaport = 8080 # port 번호
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.connect((javaip, javaport))
#
# print(" java 서버와 연결 완료 ")

print(" ---------------서버 입니다 ----------------")
def send_msg(msg):
    data = msg.encode()
    length = len(data)
    conn.sendall(length.to_bytes(4, byteorder="little"))
    conn.sendall(data)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


try:
    while True:
        # 프레임 수신
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")  # 직렬화되어 있는 binary file로 부터 객체로 역직렬화
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # 프레임 디코딩
        image = imutils.resize(image, width=200)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray, 1)
        if len(rects) == 0 and UNRECOGNIZED_COUNTER < MAXIMUM_UNRECOGNIZED_COUNT:
            UNRECOGNIZED_COUNTER += 1

        else:
            UNRECOGNIZED_COUNTER = 0

        if UNRECOGNIZED_COUNTER >= MAXIMUM_UNRECOGNIZED_COUNT:
            UNRECOGNIZED_COUNTER = 0
            send_msg('alarm')
            now = datetime.datetime.now()
            nowTime = now.strftime('%H:%M:%S')
            print(nowTime)
            # msg = '딴짓 감지 ' + nowTime
            # testmsg = msg.encode()
            # length = len(testmsg)
            # server_socket.sendall(length.to_bytes(4, byteorder="little"))
            # server_socket.sendall(testmsg)

        for rect in rects:
            shape = predictor(img_gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            both_ear = (leftEAR + rightEAR) * 500
            # 보여주기용
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
            # cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)
            # cv2.putText(image, "EAR: {}".format(round(both_ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 0, 255), 2)
            if both_ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                EYE_CLOSED_COUNTER = 0
                send_msg("alarm")
                now = datetime.datetime.now()
                nowTime = now.strftime('%H:%M:%S')
                print(nowTime)
                # msg = '졸음 감지'
                # testmsg = msg.encode()
                # length = len(testmsg)
                # server_socket.sendall(length.to_bytes(4, byteorder="little"))
                # server_socket.sendall(testmsg)
        # 영상 출력
        # cv2.imshow('TCP_Frame_Socket', image)

        # if cv2.waitKey(1) == ord('q'):  # q를 입력하면 종료
        #     break
except Exception as e:
    print("except : ", addr)
    print(str(e))
finally:
    s.close()
    # server_socket.close()

