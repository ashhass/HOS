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
image_path = '/y/ayhassen/allmerged/fulldata/image'

annot_path = '/y/ayhassen/epick_dataset/train_annotations.json'
dst_path = '/y/ayhassen/allmerged/fulldata/finalmasks/epick/finalobjects'

file = open(annot_path)
data = json.load(file)


def construct_segments(segments, segment, mask_zeros, imageBB, image_read):
    # if segment != 'right hand' and segment != 'left hand':
    if len(mask_zeros.shape) == 3:
        mask_zeros = mask_zeros[:,:,0] 
    if len(segment) != 0: 
        polygon = segments[segment]
        for index, shape in enumerate(polygon): 
            if len(shape) != 1:
                maxim_iou = 0
                sub_map = {}
                sub_map[0] = mask_zeros
                for elements in shape:
                    mask_zero = np.zeros_like(mask_zeros)
                    new_mask = draw.polygon2mask((mask_zero.shape[1], mask_zero.shape[0]), elements) 
                    new_mask = np.asarray(new_mask, dtype=np.uint8) 
                    image = cv2.rotate(new_mask, cv2.ROTATE_180)
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.flip(image, 1)
                    mask_final = np.add(mask_zero, image)  
                    maskBB, mask = construct_boxes(mask_final)
                    if maskBB != []:
                        #compute IOU for each box within the array
                        sub_iou = compute_iou(imageBB, maskBB)
                        if sub_iou > maxim_iou:
                            maxim_iou = sub_iou
                            sub_map[maxim_iou] = mask 
                mask_zeros = sub_map[maxim_iou] 
            else:
                    # print(shape[0])
                    new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), shape[0]) 
                    new_mask = np.asarray(new_mask, dtype=np.uint8) 
                    image = cv2.rotate(new_mask, cv2.ROTATE_180)
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.flip(image, 1)
                    mask_zeros = np.add(mask_zeros, image) 


            mask_zeros[mask_zeros==1] = 255
            # cv2.imwrite('./mask_zeros.png', mask_zeros)

    if len(np.unique(mask_zeros)) > 1:
        maskBB, mask = construct_boxes(mask_zeros)

    else:
        return None


    return maskBB, mask
        


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

    return newBB, mask 



def compute_iou(imageBB, maskBB):
    mask_width = abs(maskBB[2] - maskBB[0]) 
    mask_height = abs(imageBB[3] - imageBB[1])

    image_width = abs(imageBB[2] - imageBB[0])
    image_height = abs(imageBB[3] - imageBB[1])

    x_inter1 = max(maskBB[0], imageBB[0])
    x_inter2 = min(maskBB[2], imageBB[2])
    y_inter1 = max(maskBB[1], imageBB[1])
    y_inter2 = min(maskBB[3], imageBB[3])

    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1 

    area_inter = width_inter * height_inter
    
    area_mask = mask_width * mask_height
    area_image = image_width * image_height

    area_union = area_mask + area_image - area_inter

    iou = area_inter / area_union

    return iou

num = 1
for image_name in os.listdir(image_path):
    if image_name.startswith('EK') and (image_name[image_name.find('P') : ] in os.listdir(f'/y/ayhassen/epick_dataset/Images/train')):
        image_read = cv2.imread(f'{image_path}/{image_name}')
        label = f'{image_name}.txt'
        with open(f'{label_path}/{label}') as f:
            lines = f.readlines()
            image_arr = np.array(Image.open(f'{img_path}/{image_name}').convert('RGB'))
            for count, line in enumerate(lines):
                max_iou = 0
                #tools
                # temp = line[line.rfind('|') + 1 : ]
                
                #objects
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
                    # load segments
                    segments = data[image_name[image_name.find('P') : ]] 
                    map = {}
                    for segment in segments:
                        mask_zeros = np.zeros_like(image_arr)
                        mask_zeros = mask_zeros[:,:,0]

                        # construct segments and boxes
                        if construct_segments(segments, segment, mask_zeros, imageBB, image_read) != None:
                            maskBB, mask = construct_segments(segments, segment, mask_zeros, imageBB, image_read)

                            # compute iou
                            map[0] = mask_zeros
                            iou = compute_iou(imageBB, maskBB)
                            if iou > max_iou:
                                max_iou = iou
                                map[max_iou] = mask
                            # print(max_iou)
                    

                            # cv2.imwrite(f'/{dst_path}/{count}_{image_name[:-4]}.png', map[max_iou])
                            cv2.imwrite(f'./final_mask.png', map[max_iou])
                            cv2.imwrite(f'./image.png', image_read)
                            # cv2.imwrite(f'/{}/{count}_i{image_name[:-4]}.png', image_arr)
                            print(num)
                            num+=1
                                