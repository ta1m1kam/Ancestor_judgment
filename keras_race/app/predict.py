# 機械学習のモデルの重みを利用したWebアプリケーションにする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from datetime import datetime

from keras.preptocessing import image
from keras.models import model_from_json
import sys

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])

def upload_file():
    if reques.method == 'GET'
        return render_template('index.html')
    if request.method == 'POST'
        # アップロードされた画像を保存する
        up_file = request.files['file']
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + "jpg"
        f.save(filepath)

        # モデルを利用して判定する
        model = model_from_json(open('race_model.json').read())
        model.load_weights('../results/my_model.h5')

        ####### コメント
        # この部分でdetect_face.pyのdetect_faceメソッドを呼び出す！
        # それでできる気がする ↓↓↓↓↓↓↓resizeいらねーかも
        face_image = np.array(Image.open(filepath).resize)




if __name__ = '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
