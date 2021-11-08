from flask import Flask, render_template, request
import numpy as np
# from PIL import Image
from ctscan_images import *;
from werkzeug.utils import secure_filename
# import keras.models
# import re
import sys 
import os
# import base64
import cv2
sys.path.append(os.path.abspath("./model"))
from load import * 

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),"ctscan_images")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
global model

model = init()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index_view():
    return render_template('index.html')

# def convertImage(imgData1):
# 	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
# 	with open('output.png','wb') as output:
# 	    output.write(base64.b64decode(imgstr))

@app.route('/predict',methods=['GET','POST'])

def predict():
    if request.method == "POST":

        file = request.files['ct_scan']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img=cv2.imread('ctscan_images\\'+filename)
        print(img.shape)
        categories=["CT_COVID","CT_NonCOVID"]
        label_dict={"CT_COVID":0,"CT_NonCOVID":1}
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(100,100))
        img2=np.expand_dims(img,axis=0)
        img2=img2.reshape((1,100,100,1))
        predictions=model.predict(img2)
        category_index=model.predict_classes(img2)
        if(categories[category_index[0]]=="CT_COVID"):
            pred="Covid +ve"
        else:
            pred="Covid -ve"
    return render_template('predict.html',value = pred)



if __name__ == '__main__':
    app.run(debug=True, port=8000)