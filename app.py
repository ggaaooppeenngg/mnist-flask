from flask import Flask, render_template, request
from imageio import imread
import requests
from PIL import Image
from scipy.misc import imsave,imresize
import numpy as np
import re
import base64
import sys 
import os


app = Flask(__name__)
host = os.environ.get("INFERENCE_HOST", "http://localhost:9090")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    col = Image.open("output.png")
    col = col.resize((28,28))
    gray = col.convert('1')
    x = np.asarray(gray)
    x = np.invert(x)
    
    x = x.astype(np.float32)

    # reshape image data for use in neural network
    x = x.reshape(-1,28*28)

    payload = {"inputs":{"input:1":x.tolist()},"outputNames":["output"]}
    res = requests.post(host, json=payload)
    a = res.json()["output"][0][0]
    ret = a.index(max(a))
    return "{0}".format(ret)
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    host = os.environ.get("INFERENCE_HOST", "http://localhost:9090")
    app.run(host='0.0.0.0', port=PORT)
