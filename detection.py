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
predictor = dlib.shape_predictor(
    os.path.dirname(os.path.abspath(__file__))+"/shape_predictor_68_face_landmarks.dat")
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

def main(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    landmarks = facemark(gray)

    os.makedirs(f".{output_path.split('.')[1]}/", exist_ok=True)
    color = (255,0,0)

    for i, landmark in enumerate(landmarks):
        #新しい配列に入力画像の一部を代入
        dst = img[landmark[29][1]:landmark[32][1], landmark[17][0]:landmark[40][0]]
        #書き出し
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_left_cheek.jpg", dst)

        dst = img[landmark[29][1]:landmark[34][1], landmark[43][0]:landmark[26][0]]
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_right_cheek.jpg",dst)

        dst = img[(landmark[19][1] - (landmark[37][1]-landmark[19][1])) : landmark[24][1], landmark[19][0] : landmark[24][0]]
        cv2.imwrite(f".{output_path.split('.')[1]}/{i+1}_amount.jpg",dst)

        cv2.rectangle(img, (landmark[17][0], landmark[29][1]), (landmark[40][0], landmark[32][1]), color, thickness=2)
        cv2.rectangle(img, (landmark[43][0], landmark[29][1]), (landmark[26][0], landmark[34][1]), color, thickness=2)
        cv2.rectangle(img, (landmark[19][0], landmark[19][1] - (landmark[37][1]-landmark[19][1])), (landmark[24][0], landmark[24][1]), color, thickness=2)
    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    main("./inputs/test.jpg", "./outputs/test.jpg")