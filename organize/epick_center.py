import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import math
import shutil
import json
import imutils
from skimage import draw

img_path = '/y/ayhassen/allmerged/fulldata/image_new/allMerged5'
label_path = '/y/ayhassen/allmerged/fulldata/mask'
# image_path = '/y/ayhassen/allmerged/fulldata/image' 

annot_path = '/y/ayhassen/epick_dataset/train_annotations.json'
dst_path = '/y/ayhassen/allmerged/fulldata/finalmasks/epick/finalobjects'

file = open(annot_path)
data = json.load(file)


def check_in_box(x, y, imageBB):
    # print(x, y)
    # print(imageBB)
    if x and y:
        if x > imageBB[0] and x < imageBB[2] and y > imageBB[1] and y < imageBB[3]:
            return True

    return False 


def compute_center(maskBB):

    x_center = (maskBB[0] + maskBB[2]) / 2
    y_center = (maskBB[1] + maskBB[3]) / 2


    return x_center, y_center


def construct_boxes(mask):
    # cv2.imwrite('./temp.png', mask)
    c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(c)
    newBB = []
    if len(cnts)!= 0:
        c = max(cnts, key=cv2.contourArea)
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
        halfWidth, halfHeight = width/2, height/2
        newBB = [minpx, minpy, maxpx, maxpy] 

    # print(newBB, mask.shape)

    return newBB 



def construct_segments(segments, segment, mask_zeros, imageBB, mask_map):
    if len(mask_zeros.shape) == 3:
        mask_zeros = mask_zeros[:,:,0] 
    if len(segment) != 0: 
        polygon = segments[segment]
        for index, shape in enumerate(polygon): 
            if len(shape) != 1:
                result = False
                for elements in shape:
                    print(segment, len(elements))
                    new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), elements) 
                    new_mask = np.asarray(new_mask, dtype=np.uint8) 
                    image = cv2.rotate(new_mask, cv2.ROTATE_180)
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.flip(image, 1)
                    mask_zeros = np.add(mask_zeros, image)  
                    maskBB = construct_boxes(mask_zeros) 

                    x_center, y_center = compute_center(maskBB) 
                    result = check_in_box(x_center, y_center, imageBB)  

                    if result == True:
                        mask_map[segment] = mask_zeros
                    
            else:
                    new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), shape[0]) 
                    new_mask = np.asarray(new_mask, dtype=np.uint8) 
                    image = cv2.rotate(new_mask, cv2.ROTATE_180)
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.flip(image, 1)
                    mask_zeros = np.add(mask_zeros, image) 
                    maskBB = construct_boxes(mask_zeros)
                    x_center, y_center = compute_center(maskBB) 
                    result = check_in_box(x_center, y_center, imageBB)

                    if result == True:
                        mask_map[segment] = mask_zeros

    return mask_map
        


num = 1
image_name = 'EK_0034_P25_107_frame_0000062963.jpg' 
# for image_name in os.listdir(img_path):
#     if image_name.startswith('EK') and (image_name[image_name.find('P') : ] in os.listdir(f'/y/ayhassen/epick_dataset/Images/train')):
label = f'{image_name}.txt'
with open(f'{label_path}/{label}') as f:
    lines = f.readlines()
    image_arr = np.array(Image.open(f'{img_path}/{image_name}').convert('RGB'))

    for count, line in enumerate(lines):
        split_frwrd = line[line.find('|') + 1 : ]
        split_frwrd2 = split_frwrd[split_frwrd.find('|') + 1 : ]
        temp1 = split_frwrd2[split_frwrd2.find('|') + 1: ]
        split_bck = temp1[ : temp1.rfind('|') - 1]
        temp = split_bck[ : split_bck.rfind('|') - 1]

        if temp.strip() != 'None': 
            bbox = temp.split(",")
            zero = int(float(bbox[0].strip()))
            one = int(float(bbox[1].strip()))
            two = int(float(bbox[2].strip()))
            three = int(float(bbox[3].strip()))

            imageBB = [zero, one, two, three]
            cv2.rectangle(image_arr, (zero, one), (two, three), (0,255,0), 2)
            plt.imsave('./image.jpg', image_arr)

            segments = data[image_name[image_name.find('P') : ]] 
            # print(segments.keys())
            # for element in segments.keys():
            #     print(element, len(segments[element][0])) 
            mask_zeros = np.zeros_like(image_arr)
            mask_zeros = mask_zeros[:,:,0]
            mask_final = np.zeros_like(mask_zeros)
            mask = np.zeros_like(mask_zeros)
            mask_map = {}
            for segment in segments:
                mask_array = construct_segments(segments, segment, mask_zeros, imageBB, mask_map) 

            print(mask_array.keys())
            
                
            

            # mask_final[mask_final>=1] = 255
            # cv2.imwrite('./mask.png', mask_final)
            # cv2.imwrite(f'./{image_name}', image_arr)
                # print(box)

        sys.exit()
