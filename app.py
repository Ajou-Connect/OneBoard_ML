#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, Response,request  ## flask 라이브러리에서 Flask import
import numpy as np
import imutils
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
app = Flask(__name__)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/')
def index():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    MINIMUM_EAR = 240
    MAXIMUM_FRAME_COUNT = 30
    MAXIMUM_UNRECOGNIZED_COUNT = 60
    EYE_CLOSED_COUNTER = 0
    UNRECOGNIZED_COUNTER = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_2.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    vid_in = cv2.VideoCapture(0)

    while True:
        ret, image_o = vid_in.read()
        image = imutils.resize(image_o, width = 1000)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray,0)

        if len(rects) == 0:
            UNRECOGNIZED_COUNTER += 1
            #cv2.putText(image, "CANT FIND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            UNRECOGNIZED_COUNTER = 0

        if UNRECOGNIZED_COUNTER >= MAXIMUM_UNRECOGNIZED_COUNT:
            cv2.putText(image, "CAMERA!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        for rect in rects:
            shape = predictor(img_gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            both_ear = (leftEAR + rightEAR) * 500

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(image, [leftEyeHull], -1, (255,0,0), 2)
            cv2.drawContours(image, [rightEyeHull], -1, (255,0,0), 2)

            if both_ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            cv2.putText(image, "EAR: {}".format(round(both_ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        key = cv2.waitKey(1)
        #esc,
        if key == 27:
            break

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#    cv2.destroyAllWindows()
if __name__ == "__main__":
    app.run(debug = True)


# In[ ]:




