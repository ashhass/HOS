import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import math
import shutil
num = 1
numbre = 1
src_path_img = '/y/ayhassen/allmerged/fulldata/image_new/allMerged5' 
label_path = '/y/ayhassen/allmerged/fulldata/mask'
final_mask = '/y/ayhassen/allmerged/fulldata/finalmasks/masks'
image_path = './missingCCobjects'
dst_path = '/y/ayhassen/allmerged/fulldata/finalmasks/masksv2.0'

for image_name in os.listdir(image_path):
    label = f'{image_name}.txt'
    with open(f'{label_path}/{label}') as f:
        lines = f.readlines()
        image_arr = np.array(Image.open(f'{image_path}/{image_name}').convert('RGB'))
        for count, line in enumerate(lines):
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

                # print(two - zero, three - one)
                # cv2.rectangle(image_arr, (zero, one), (two , three), (255, 0, 0), 2)
                # plt.imsave('./coco_image.jpg', image_arr) 
                # image = image_arr[one : three, zero : two]
                # cv2.imwrite('./coco_crop.jpg', image) 
                
                image_crop = image_arr[one : three, zero : two]
                if image_crop.shape[0] != 0 and image_crop.shape[1] != 0:
                    cv2.imwrite(f'/y/ayhassen/allmerged/CC/CCmissing/image_crops/{count}_{image_name}', image_crop)
                    print(num)
                    num+=1

                else:
                    mask_zeros = np.zeros_like(image_arr)
                    cv2.imwrite(f'{dst_path}/3_{count}_{image_name[:-4]}.png', mask_zeros)
                    # print(numbre)
                    numbre+=1

            print(f'Number of black masks : {numbre}')
            print(f'Number of normal masks : {num}') 