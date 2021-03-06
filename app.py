from flask import Flask, Response, request, render_template, flash, url_for
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
def index():
    return render_template('index.html')

@app.route('/facedetection')
def facedetection():
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
            detect = 'test'
            sound()
            

def Send(socket, camera, encode_param):
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
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    t1 = threading.Thread(target=Send, args= (client_socket, camera, encode_param,))
    t2 = threading.Thread(target=Recv, args= (client_socket,))
    t2.daemon = True
    t1.daemon = True
    t2.start()
    t1.start()

    while True:
        ret, image_o = camera.read()
        ret, buffer = cv2.imencode('.jpg', image_o, encode_param)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug=False, threaded=True)
    # app.run(debug=True, threaded=True)
# In[ ]:
