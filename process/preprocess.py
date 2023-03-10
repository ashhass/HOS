import os
import cv2
from skimage import draw
import numpy as np
import imutils
import matplotlib.pyplot as plt
import json
from PIL import Image
import sys
import scipy.ndimage



train_image_path = "/y/ayhassen/epick_dataset/Images/train"
val_image_path = "/y/ayhassen/epick_dataset/Images/val"

train_mask_path = "/y/ayhassen/epick_dataset/train_annotations.json"
val_mask_path = "/y/ayhassen/epick_dataset/val_annotations.json"
hos_path = "/y/relh/epickitchens/Annotations/hos/hos_trainval.json"


file = open(train_mask_path)
data = json.load(file)

count = 0
for element in data:
    image_str = os.path.join(train_image_path, element)
    true_image = np.array(Image.open(image_str).convert('RGB'))
    for index, object in enumerate(data[element]):
        if object != 'right hand' and object != 'left hand':
            for segment in data[element][object]:
                if len(segment) == 1:
                    mask_zeros = np.zeros_like(true_image[:,:,0]) 
                    new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), segment[0]) 
                    new_mask = np.asarray(new_mask, dtype=np.uint8) 
                    image = cv2.rotate(new_mask, cv2.ROTATE_180)
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.flip(image, 1) 
                    mask_zeros = np.add(mask_zeros, image)
                for indx, x in enumerate(mask_zeros):
                    for indice, y in enumerate(x):
                        if y >= 1:
                            mask_zeros[indx][indice] = 1
                else:
                    mask_zeros = np.zeros_like(true_image[:,:,0]) 
                    for part in segment:
                        new_mask = draw.polygon2mask((mask_zeros.shape[1], mask_zeros.shape[0]), part) 
                        new_mask = np.asarray(new_mask, dtype=np.uint8) 
                        image = cv2.rotate(new_mask, cv2.ROTATE_180)
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        image = cv2.flip(image, 1) 
                        mask_zeros = np.add(mask_zeros, image)
                for indx, x in enumerate(mask_zeros):
                    for indice, y in enumerate(x):
                        if y >= 1:
                            mask_zeros[indx][indice] = 1

            #----------PAD IMAGE-------------#
            pad_len = 500
            image_borderType = cv2.BORDER_REPLICATE
            mask_borderType = cv2.BORDER_CONSTANT
            image = cv2.copyMakeBorder(true_image, pad_len, pad_len, pad_len, pad_len, image_borderType, None)
            mask_zeros = cv2.copyMakeBorder(mask_zeros, pad_len, pad_len, pad_len, pad_len, mask_borderType, None)
            
            
            #-----------COMPUTE_HANDS_CENTER---------#
            if len(np.unique(mask_zeros)) >= 2 :
                x, y = np.meshgrid(np.arange(image.shape[1]).astype(float), np.arange(image.shape[0]).astype(float))
                center_x = np.mean(x[mask_zeros==1])
                center_y = np.mean(y[mask_zeros==1])
                image = cv2.circle(image, (int(center_x), int(center_y)), radius=0, color=(0, 255, 0), thickness=10)
                # contours, hierarchy = cv2.findContours(mask_zeros, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # img_contours = image
                # cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
                for idx, randomCrops in enumerate(range(10)):
                    noiseX = int(np.random.normal()*pad_len*0.1)
                    noiseY = int(np.random.normal()*pad_len*0.1) 
                    bbox = [(int((center_x - noiseX) - (pad_len - noiseX)), int((center_y - noiseY) + (pad_len - noiseY))), (int((center_x - noiseX) - (pad_len - noiseX)), int((center_y - noiseY) - (pad_len - noiseY))), (int((center_x - noiseX) + (pad_len - noiseX)), int((center_y - noiseY) + (pad_len - noiseY))), (int((center_x - noiseX) + (pad_len - noiseX)), int((center_y- noiseY) - (pad_len - noiseY)))]                                    
                    image_crop = image[bbox[1][1] : bbox[0][1], bbox[0][0] : bbox[2][0]]
                    mask_crop = mask_zeros[bbox[1][1] : bbox[0][1], bbox[0][0] : bbox[2][0]]

                    plt.imsave(f'/y/ayhassen/epick_object_crop/Images/active_objects_updated/train/{element}_{index}_{idx}.jpg', image_crop)   
                    plt.imsave(f'/y/ayhassen/epick_object_crop/Masks/active_objects_updated/train/mask_{element}_{index}_{idx}.jpg', mask_crop, cmap = 'gray')  

                    # plt.imsave(f'../image/image_{element}_{index}_{idx}.jpg', image_crop)  
                    # plt.imsave(f'../mask/mask_{element}_{index}_{idx}.jpg', mask_crop)                         

                    print(count)
                    count+=1
