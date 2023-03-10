from ctypes.wintypes import MAX_PATH
import math
import os
import sys
import cv2
import imutils
import matplotlib.pyplot as plt
import shutil
import json
import numpy as np
from skimage import draw
import torch
from PIL import Image

#-----------ORIGIN PATHS--------------#
train_mask_path = "/y/relh/epickitchens/Annotations/train"
val_mask_path = "/y/relh/epickitchens/Annotations/val"
train_image_path = "/y/ayhassen/epick_dataset/Images/train"
val_image_path = "/y/ayhassen/epick_dataset/Images/val"


def bounding_box(points):

    bot_x = min(point[0] for point in points)
    bot_y = min(point[1] for point in points)
    top_x = max(point[0] for point in points)
    top_y = max(point[1] for point in points)


    return [bot_x, bot_y, top_x, top_y]

count = 0
maxheight, maxwidth = 0, 0
for elements in os.listdir(train_mask_path): 
    file = open(os.path.join(train_mask_path, elements))
    data = json.load(file) 
    for idx, element in enumerate(data):
        image_name = data[idx]['documents'][0]['name'] 
        image_str = os.path.join(train_image_path, image_name) 
        true_image = cv2.imread(image_str)
        cont = data[idx]['annotation']['annotationGroups'][0]['annotationEntities']
        for index, elements in enumerate(cont):
            # if cont[index]['name'] != "left hand" and cont[index]['name'] != "right hand":
                segments = cont[index]['annotationBlocks'][0]['annotations'][0]['segments']
                mask_zeros = np.zeros_like(true_image)
                for segment in segments:
                    if len(segment) != 0:
                        new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), segment) 
                        new_mask = np.asarray(new_mask, dtype=np.uint8) 
                        image = cv2.rotate(new_mask, cv2.ROTATE_180)
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        image = cv2.flip(image, 1)
                        if len(mask_zeros.shape) == 3:
                            mask_zeros = np.add(mask_zeros[:,:,0], image)
                        else:
                            mask_zeros = np.add(mask_zeros, image) 
                for indx, x in enumerate(mask_zeros):
                    for indice, y in enumerate(x):
                        if y >= 1:
                            mask_zeros[indx][indice] = 1

                box = bounding_box(mask_zeros) 
                c = cv2.findContours(mask_zeros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(c)
                if len(cnts)!= 0:
                    c = max(cnts, key=cv2.contourArea)
                else:
                    continue
                px = []
                py = [] 
                imgx = []
                imgy = []
                for x in c[:, 0]:
                    px.append(x[0])
                    py.append(x[1]) 
                maxpx = max(px)
                minpx = min(px)
                maxpy = max(py)
                minpy = min(py)
                width = maxpx - minpx
                height = maxpy - minpy
                if width > maxwidth:
                    maxwidth = width
                    pad_width = maxwidth
                if height > maxheight:
                    maxheight = height
                    pad_height = maxheight
                halfWidth, halfHeight = width/2, height/2
                newBB = [minpx, minpy, maxpx, maxpy] 
                if newBB[3] < maxpy and newBB[2] < maxpx or newBB[0] >= minpx and newBB[1] >= minpy:
                    final_mask = mask_zeros[newBB[1] : newBB[3], newBB[0] : newBB[2]]
                    for idx, randomCrops in enumerate(range(10)):
                        noiseMinX = int(np.random.normal()*width*0.1)
                        noiseMinY = int(np.random.normal()*height*0.1) 
                        noiseMaxX = int(np.random.normal()*width*0.1)
                        noiseMaxY = int(np.random.normal()*height*0.1) 
                        newBB = [minpx-noiseMinX, minpy-noiseMinY, maxpx+noiseMaxX, maxpy+noiseMaxY]
                        final_image = true_image[newBB[1] : newBB[3], newBB[0] : newBB[2]]
                        final_mask = mask_zeros[newBB[1] : newBB[3], newBB[0] : newBB[2]]
                        # borderType = cv2.BORDER_REPLICATE 
                        mask_borderType = cv2.BORDER_CONSTANT
                        pad_width = (500 - final_image.shape[0]) // 2
                        pad_height = (500 - final_image.shape[1]) // 2
                        mask_repl = cv2.copyMakeBorder(final_mask, abs(pad_height), abs(pad_height), abs(pad_width), abs(pad_width), mask_borderType, None)
                        image_repl = cv2.copyMakeBorder(final_image, abs(pad_height), abs(pad_height), abs(pad_width), abs(pad_width), mask_borderType, None, value = (128, 128, 128))

                        if(len(np.unique(final_mask))) >= 2:
                            # cv2.imwrite(f'/y/ayhassen/epick_object_crop/Images/crop_fit/train/{image_name}_{index}_{idx}.jpg', image_repl)
                            # plt.imsave(f'/y/ayhassen/epick_object_crop/Masks/crop_fit/train/{image_name}_{index}_{idx}.jpg', mask_repl, cmap="gray") 
                            count+=1
                            print(count)