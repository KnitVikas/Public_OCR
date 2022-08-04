#server side code
from flask import Flask, render_template, request, send_file,jsonify
import cv2
import numpy as np
import base64
import io
import PIL
from PIL import Image
from test import OCRING, load_TD_model
from demo import load_model
net = load_TD_model()


model = load_model()
app = Flask(__name__)
@app.route('/manipulate_image', methods=['GET', 'POST'])
def image_manipulation():

    photo = request.get_json()['user_photo']
    photo_data = base64.b64decode(photo)
    file_like = io.BytesIO(photo_data)
    img = PIL.Image.open(file_like)
    img  = img.convert("RGB")
    #convert to cv2 format
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    img, results = OCRING(model,net,open_cv_image)
    img = base64.b64encode(img).decode('utf-8')
    return jsonify({
                'msg': 'success', 
                'img': img,
                'results': results
           })

if __name__ == '__main__':
    app.run(debug=True, port=8000)