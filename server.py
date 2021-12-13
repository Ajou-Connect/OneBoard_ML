import socket
import cv2
import pickle
import struct
import imutils
import cv2
import dlib
import time
import datetime
import threading
from imutils import face_utils
from scipy.spatial import distance as dist


MINIMUM_EAR = 300
MAXIMUM_FRAME_COUNT = 50
MAXIMUM_UNRECOGNIZED_COUNT = 100
EYE_CLOSED_COUNTER = 0
UNRECOGNIZED_COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_2.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print(" \n꺼지지 않게 하려면 nohup python3 server.py& 로 실행해야함")
print(" \n종료하려면 ps -ef | grep server.py 후 kill pid 로 종료\n")


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

while True:
    ip = '0.0.0.0'
    port = 8090
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(10)

    print('클라이언트 연결 대기\n')
    conn, addr = s.accept()
    print(addr)

    data = b""
    payload_size = struct.calcsize(">L")

    print('Connected by', addr)
    
    # print(" 자바 서버와 연결하는 코드는 현재 주석처리중 ")
    javaip = '115.85.182.194' # ip 주소
    javaport = 8080 # port 번호
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((javaip, javaport))
    print(" java 서버 포트번호 8080 와 연결 완료 \n\n\n")

    print(" 연결 완료 ! \n")
    print(" ---------------서버 입니다 ----------------\n")
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
                try:
                    msg = '딴짓 감지 ' + nowTime
                    testmsg = msg.encode()
                    length = len(testmsg)
                    server_socket.sendall(length.to_bytes(4, byteorder="little"))
                    server_socket.sendall(testmsg)
                except:
                    print("java 에 데이터를 보내는것에서 오류")

            for rect in rects:
                shape = predictor(img_gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                both_ear = (leftEAR + rightEAR) * 500

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
                    try:
                        msg = '딴짓 감지 ' + nowTime
                        testmsg = msg.encode()
                        length = len(testmsg)
                        server_socket.sendall(length.to_bytes(4, byteorder="little"))
                        server_socket.sendall(testmsg)
                    except:
                        print("java 에 데이터를 보내는것에서 오류")

    except Exception as e:
        print("except : ", addr)
        print(str(e))
    finally:
        print(" 연결 종료 !! ")
        s.close()
        server_socket.close()

