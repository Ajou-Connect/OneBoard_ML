import cv2
import socket
import struct
import pickle
import time
import threading
import winsound as sd


def Send(socket):
    while True:
        ret, frame = camera.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame, 0)
        size = len(data)
        socket.sendall(struct.pack(">L", size) + data)


def Recv(socket):
    while True:
        get_data = socket.recv(4)
        length = int.from_bytes(get_data, "little")
        get_data = socket.recv(length)
        msg = get_data.decode()
        print('Received from : ', msg)
        if msg == 'alarm':
            beepsound()

def beepsound():
    fr = 2000
    du = 1000
    sd.Beep(fr, du)

ip = '115.85.182.194' # ip 주소
port = 8090 # port 번호

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))
print('연결 성공')

print("----------클라이언트 입니다 ------------------")
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


t1 = threading.Thread(target=Send, args= (client_socket,))
t2 = threading.Thread(target=Recv, args= (client_socket,))

t2.start()
t1.start()
