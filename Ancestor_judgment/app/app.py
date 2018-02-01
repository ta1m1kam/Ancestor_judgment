# 機械学習のモデルの重みを利用したWebアプリケーションにする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
# from PIL import Image
import cv2
from datetime import datetime
# import matplotlib.pyplot as plt

# from keras.preprocessing import image
from keras.models import model_from_json, load_model
import sys
# import predict_race


def detect_face(image):
    print(image.shape)
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(64,64))
            img = np.expand_dims(img,axis=0)
            name = detect_who(img)
    #顔が検出されなかった時
    else:
        name = "なし"

    return name

def detect_who(img):
    ## TODO リファクタリンフ
    model = load_model('../results/my_model.h5')
    nameNumLabel=np.argmax(model.predict(img))
    if nameNumLabel== 0:
        name="Asian"
    elif nameNumLabel==1:
        name="Caucasian"
    elif nameNumLabel==2:
        name="Hispanic"
    elif nameNumLabel==3:
        name="Multiracial"
    elif nameNumLabel==4:
        name="Black"
    print(name)
    return name

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])

def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # アップロードされた画像を保存する
        up_file = request.files['file']
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
        up_file.save(filepath)

        ####### コメント
        # この部分でdetect_face.pyのdetect_faceメソッドを呼び出す！
        # それでできる気がする ↓↓↓↓↓↓↓resizeいらねーかも
        face_image = cv2.imread(filepath)
        if face_image is None:
            print("Not open:")
        b,g,r = cv2.split(face_image)
        face_image = cv2.merge([r,g,b])

        predict = detect_face(face_image)
        return render_template('index.html', filepath = filepath , predict = predict )



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
