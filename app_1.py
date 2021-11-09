#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, Response, request, jsonify  ## flask 라이브러리에서 Flask import
import imutils
import cv2
import dlib
import datetime
import time
from imutils import face_utils
from scipy.spatial import distance as dist

app = Flask(__name__)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/', methods=['GET', 'POST'])
def gen_frames():
    MINIMUM_EAR = 300
    MAXIMUM_FRAME_COUNT = 4
    MAXIMUM_UNRECOGNIZED_COUNT = 20
    EYE_CLOSED_COUNTER = 0
    UNRECOGNIZED_COUNTER = 0
    ts_list = []
    type_list = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_2.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vid_in = cv2.VideoCapture(0)
    # vid_in = cv2.VideoCapture("*.mp4"ㅡ)
    # while True:
    ### Test 용 while 문 ###
    ### 10초동안만 영상 감지를함. ###
    t_end = time.time() + 10
    while time.time() < t_end:
    #######################
        facedetection = {'detected_type': type_list,
                         'timestamp': ts_list,
                         }

        ret, image_o = vid_in.read()
        ts = datetime.datetime.now().timestamp()
        date = datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')
        image = imutils.resize(image_o, width=500)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray, 1)

        if len(rects) == 0 and UNRECOGNIZED_COUNTER < MAXIMUM_UNRECOGNIZED_COUNT:
            UNRECOGNIZED_COUNTER += 1
        else:
            UNRECOGNIZED_COUNTER = 0

        if UNRECOGNIZED_COUNTER >= MAXIMUM_UNRECOGNIZED_COUNT:
            EYE_CLOSED_COUNTER = 0
            ts_list.append(date)
            type_list.append("undetected")

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
            cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
            cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)

            if both_ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                ts_list.append(date)
                type_list.append("Drowsiness")
                EYE_CLOSED_COUNTER = 0

    return jsonify(facedetection)

if __name__ == "__main__":
    app.run(debug=True)

# In[ ]:
