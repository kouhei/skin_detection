"""
dlibとopenCV使って顔器官検出
https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
"""

import cv2
import dlib
import numpy as np
import os

# Cascade files directory path
CASCADE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/"
LEARNED_MODEL_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/learned-models/"
predictor = dlib.shape_predictor(
    LEARNED_MODEL_PATH + 'helen-dataset.dat')
predictor = dlib.shape_predictor(
    os.path.dirname(os.path.abspath(__file__))+"/shape_predictor_68_face_landmarks.dat"
)
face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH + 'haarcascade_frontalface_default.xml')


def face_position(gray_img):
    """Detect faces position
    Return:
        faces: faces position list (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(gray_img, minSize=(100, 100))
    return faces


def facemark(gray_img):
    """Recoginize face landmark position by i-bug 300-w dataset
    Return:
        landmarks = [
        [x, y],
        [x, y],
        ...
        ]
        [0~40]: chin
        [41~57]: nose
        [58~85]: outside of lips
        [86-113]: inside of lips
        [114-133]: right eye
        [134-153]: left eye
        [154-173]: right eyebrows
        [174-193]: left eyebrows
    """
    faces_roi = face_position(gray_img)
    landmarks = []

    for face in faces_roi:
        x, y, w, h = face
        face_img = gray_img[y: y + h, x: x + w];

        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_img, 1)

        landmarks = []
        for rect in rects:
            landmarks.append(np.array([[p.x, p.y] for p in predictor(gray_img, rect).parts()]))
    return landmarks

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
        # _, frame = cap.read()
    image_path = "./inputs/test.jpg"
    output_path = "./outputs/test.jpg"
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    landmarks = facemark(gray)

    color = (255,0,0)
    font = cv2.FONT_HERSHEY_PLAIN
    for i, landmark in enumerate(landmarks):
        # for i, points in enumerate(landmark):
            # cv2.drawMarker(frame, (points[0], points[1]), (21, 255, 12))
            #文字の書き込み
            # cv2.putText(frame,str(i),(points[0],points[1]),font, 1,(255,255,0))

        #新しい配列に入力画像の一部を代入
        dst = frame[landmark[29][1]:landmark[32][1], landmark[17][0]:landmark[40][0]]
        #書き出し
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_left_cheek.jpg", dst)

        dst = frame[landmark[29][1]:landmark[34][1], landmark[43][0]:landmark[26][0]]
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_right_cheek.jpg",dst)

        dst = frame[(landmark[19][1] - (landmark[37][1]-landmark[19][1])) : landmark[24][1], landmark[19][0] : landmark[24][0]]
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_amount.jpg",dst)

        cv2.rectangle(frame, (landmark[17][0], landmark[29][1]), (landmark[40][0], landmark[32][1]), color, thickness=2)
        cv2.rectangle(frame, (landmark[43][0], landmark[29][1]), (landmark[26][0], landmark[34][1]), color, thickness=2)
        cv2.rectangle(frame, (landmark[19][0], landmark[19][1] - (landmark[37][1]-landmark[19][1])), (landmark[24][0], landmark[24][1]), color, thickness=2)
    cv2.imwrite(output_path, frame)
        # cv2.imshow("video frame", frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()