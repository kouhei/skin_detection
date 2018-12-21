import cv2, sys, os
"""
(画像を検出した範囲で切り抜く)
顔と目とかを検出

分類器ディレクトリ(以下から取得)
https://github.com/opencv/opencv/blob/master/data/haarcascades/
https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/
"""

INPUT_PATH = os.path.dirname(os.path.abspath(__file__))+"/inputs/"
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))+"/outputs/"
# 学習済モデルファイル
print(os.path.dirname(os.path.abspath(__file__)))
models = {
    "default":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_frontalface_default.xml",
    "alt":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_frontalface_alt.xml",
    "alt2":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_frontalface_alt2.xml",
    "tree":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_frontalface_alt_tree.xml",
    "profile":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_profileface.xml",
    "nose":os.path.dirname(os.path.abspath(__file__))+"/models/Nariz.xml",
    "eyes":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_eye.xml",
    "left_eye":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_lefteye_2splits.xml",
    "right_eye":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_righteye_2splits.xml",
    "profile":os.path.dirname(os.path.abspath(__file__))+"/models/haarcascade_profileface.xml",
}

def detect(model_name, img, color, detect_args={}):
    args = {"scaleFactor":1.3, "minNeighbors":2, "minSize":(5, 5)}
    model = models[model_name]
    cascade = cv2.CascadeClassifier(model)
    for arg in args:
        if arg in detect_args:
            args[arg] = detect_args[arg]
    detected = cascade.detectMultiScale(img, **args)
    if len(detected) > 0:
        for rect in detected:
            cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
            return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    else:
        print(model_name,"isn't detected:", file_name)
        return img
        # raise Exception(model_name+" isn't detected: "+file)

# 直接実行されている場合に通る(importされて実行時は通らない)
if __name__ == "__main__":
    # 画像ファイル読込
    # file = "test.jpg"
    # file = "hayano.jpg"
    file = "hayano2.jpg"
    file_name = file.split(".")[0]
    img = cv2.imread(INPUT_PATH + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model_name = "default"
    face = detect(model_name, img, (0, 0, 255), {"scaleFactor":10, "minNeighbors":2, "minSize":(5, 5)})
    left_eye = detect("left_eye", face, (255,0,0))
    right_eye = detect("right_eye", face, (255,0,0))
    nose = detect("nose", face, (255,0,0))
    cv2.imwrite(OUTPUT_PATH + file_name+"_"+model_name+"_eyes.jpg", img)