from flask import Flask, Response, request  ## flask 라이브러리에서 Flask import
import imutils
import cv2
import socket
import pickle
import struct
import threading
import pygame

app = Flask(__name__)
pygame.init()

@app.route('/')
def home():
    return 'hello this is ML page'


@app.route('/facedetection')
def index():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def sound():
    soundObj = pygame.mixer.Sound('beep.wav')  # 사운드 파일을 로딩한다
    soundObj.play()

def Recv(socket):
    while True:
        get_data = socket.recv(4)
        length = int.from_bytes(get_data, "little")
        get_data = socket.recv(length)
        msg = get_data.decode()
        print('Received from : ', msg)
        if msg == 'alarm':
            sound()

def Send(socket, camera, encode_param):
    print(camera)
    while True:
        ret, image_o = camera.read()
        ret, buffer = cv2.imencode('.jpg', image_o, encode_param)
        data = pickle.dumps(buffer, 0)
        size = len(data)
        socket.sendall(struct.pack(">L", size) + data)

def gen_frames():
    ip = '115.85.182.194'  # ip 주소 115.85.182.194
    port = 8090  # port
    camera = cv2.VideoCapture(0)
    if camera is None or not camera.isOpened():
        print('Warning: unable to open video source')
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    t1 = threading.Thread(target=Send, args= (client_socket, camera, encode_param))
    t2 = threading.Thread(target=Recv, args= (client_socket,))
    # t2.daemon = True
    t2.start()
    t1.start()

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8100, debug=False, threaded=True)

# In[ ]:
