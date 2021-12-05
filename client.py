import cv2
import socket
import struct
import pickle
import time
import threading

def Send(socket):
    while True:
        ret, frame = camera.read()  # 카메라 프레임 읽기
        result, frame = cv2.imencode('.jpg', frame, encode_param)  # 프레임 인코딩
        # 직렬화(serialization) : 효율적으로 저장하거나 스트림으로 전송할 때 객체의 데이터를 줄로 세워 저장하는 것
        # binary file : 컴퓨터 저장과 처리 목적을 위해 이진 형식으로 인코딩된 데이터를 포함
        data = pickle.dumps(frame, 0)  # 프레임을 직렬화화하여 binary file로 변환
        size = len(data)
        # print("Frame Size : ", size) # 프레임 크기 출력

        # 데이터(프레임) 전송
        socket.sendall(struct.pack(">L", size) + data)


def Recv(socket):
    while True:
        get_data = socket.recv(4)
        length = int.from_bytes(get_data, "little")
        # 데이터 길이를 받는다.
        get_data = socket.recv(length)
        # 데이터를 수신한다.
        # msg = get_data.decode()
        # 데이터를 출력한다
        tm = time.localtime(time.time())
        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        print('Received time: ' + string + '     info' + repr(get_data.decode()))
        # print('Received from : ', msg)

ip = 'localhost' # ip 주소
port = 20001 # port 번호

# 소켓 객체를 생성 및 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))
print('연결 성공')

# 카메라 선택
camera = cv2.VideoCapture(0)

# 크기 지정
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200) # 가로
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200) # 세로

# 인코드 파라미터
# jpg의 경우 cv2.IMWRITE_JPEG_QUALITY를 이용하여 이미지의 품질을 설정
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


t1 = threading.Thread(target=Send, args= (client_socket,))
t2 = threading.Thread(target=Recv, args= (client_socket,))

t2.start()
t1.start()
