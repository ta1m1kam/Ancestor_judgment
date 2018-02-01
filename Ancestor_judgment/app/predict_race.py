import numpy as np
import cv2
from keras.models import model_from_json, load_model

# モデルを利用して判定する
model = model_from_json(open('../results/race_model.json').read())
model.load_weights('../results/my_model.h5')

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
            img=np.expand_dims(img,axis=0)
            name = detect_who(img)
            # cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

    #顔が検出されなかった時
    else:
        name = "no face"

    return name

def detect_who(img):
    #予測
    name=""
    # print(model.predict(img))
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
