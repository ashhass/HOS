import os
import numpy as np
import cv2
from PIL import Image
import shutil


mask_path = '/y/ayhassen/allmerged/fulldata/finalmasks/masksv1.0'
image_path = '/y/ayhassen/allmerged/fulldata/objects/image'
num = 0

for mask in os.listdir(mask_path):
    if mask.split('_')[2] == 'AR' and (mask.startswith('3') or mask.startswith('4')):
        # print(mask.split('_'))
        mask_read = np.array(Image.open(f'{mask_path}/{mask}'))
        if len(np.unique(mask_read)) == 1 and mask_read.shape[0] <= 500 and mask_read.shape[1] <= 500:
            # shutil.copy2(f'{image_path}/{mask.split("_")[1]}_{mask[mask.find("ND"): ].replace("png", "jpg")}', '/y/ayhassen/allmerged/analysis/objects/ND')
            print(num)
            num+=1

