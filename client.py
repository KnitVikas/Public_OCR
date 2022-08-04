# import argparse
import requests
import base64
import json
import sys
import matplotlib.pyplot as plt
from config import get_args
import io
args = get_args(sys.argv[1:])
import PIL
import cv2
URL = "http://127.0.0.1:8000/manipulate_image"

#  first, encode our image with base64
with open("/home/megh/Desktop/Website_project/Image_manipulation/images/Lobby 75.PNG", "rb") as imageFile:
    img = base64.b64encode(imageFile.read()).decode('utf-8')
response = requests.post(url = URL, json={"user_photo":img})
json_data = json.loads(response.content)
print(json_data["results"])
print(json_data["img"])

#decode byte image
# photo_data = base64.b64decode(json_data["img"])
# file_like = io.BytesIO(photo_data)
# img = PIL.Image.open(json_data["results"][0])
# img  = img.convert("RGB")
# cv2.imshow("recieved",img)
# cv2.waitKey(0)
