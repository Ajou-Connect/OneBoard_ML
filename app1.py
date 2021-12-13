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


def gen_frames():
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

    while True:
        ret, image_o = camera.read()
        ret, buffer = cv2.imencode('.jpg', image_o)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8100, debug=False)

# In[ ]:
