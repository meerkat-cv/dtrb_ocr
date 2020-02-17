import sys
import os

import cv2
sys.path.insert(0, "..")
from dtrb_ocr import DTRB_OCR

ocr = DTRB_OCR('../../gorc/data/dtrb_ocr/TPS-ResNet-BiLSTM-Attn.pth',
        '0123456789abcdefghijklmnopqrstuvwxyz/:,.()-',
        imgW=100, use_gpu=True)

for im_name in sorted(os.listdir('../demo_image/')):
    if os.path.splitext(im_name)[-1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
        im = cv2.imread('../demo_image/'+im_name)
        print(im_name, ocr.ocr_word(im))
