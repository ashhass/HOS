import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import sys



annot_path = "epick_hands.json"
mask_path = '/y/ayhassen/epick_dataset/Masks/train/hands_updated'


file = open(f'{annot_path}')
data = json.load(file)

for image in os.listdir(f'{mask_path}'):
    image_read = cv2.imread(f'{mask_path}/{image}')
    mask = np.zeros(image_read.shape)
    for block in data["bbox"]:
        if f'{image[:image.rfind("_")]}.jpg' in block:
            point = block[f'{image[:image.rfind("_")]}.jpg']
            mask[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] = image_read[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])]    
            cv2.imwrite('image.jpg', mask)
