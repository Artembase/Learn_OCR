import pytesseract
import cv2
import re
from pytesseract import Output
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

img = cv2.imread('kj.jpg')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def convert_grayscale(img_convert_grayscale):
    print('1')
    img = cv2.cvtColor(img_convert_grayscale, cv2.COLOR_BGR2GRAY)
    return img


def blur(img_blur, param):
    img_blur = cv2.medianBlur(img_blur, param)
    print('2')
    return img_blur


def threshold(img_threshold):
    img_threshold = cv2.threshold(img_threshold, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print('3')
    return img_threshold

#
# img1 = convert_grayscale(img)
# img2 = blur(img1, 1)
# img3 = threshold(img2)

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255,0), 2)


d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())
keys = list(d.keys())

# date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[12])/(19|20)\d\d5'

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(float(d['conf'][i])) > 60:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


custom_config = ' rus '
print(pytesseract.image_to_string(img, lang='rus'), 'nnjnjnj')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




