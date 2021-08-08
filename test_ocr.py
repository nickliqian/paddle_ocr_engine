import requests
import base64
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math

from function import cv2_to_base64, draw_server_result
import os


def req():
    is_visualize = True
    headers = {"Content-type": "application/json"}
    image_file = "img/154.jpg"
    url = "http://127.0.0.1:8868/predict/ocr_system"

    with open(image_file, 'rb') as f:
        img = f.read()

    data = {'images': [cv2_to_base64(img)]}
    r = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=6)
    res = r.json()["results"][0]
    print(res)

    if is_visualize:
        draw_img = draw_server_result(image_file, res)
        if draw_img is not None:
            draw_img_save = "./server_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == '__main__':
    req()
