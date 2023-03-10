import os 
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from PIL import Image
import numpy.ma as ma

label_path = "/y/ayhassen/allmerged/fulldata/mask" 
# mask_path = "/y/ayhassen/allmerged/fulldata/tools/mask"
# img_path = "/y/ayhassen/allmerged/fulldata/large_tool_crops/image_crops" 
mask_path = "/y/ayhassen/allmerged/CC/mask_crops" 
# mask_path = "/y/ayhassen/allmerged/fulldata/finalmasks/missing_objects" 
image_path = "/y/ayhassen/allmerged/fulldata/image_new/allMerged5"
dst_path = '/y/ayhassen/allmerged/CC/final_masks'

count = 1
num = 1

for mask in os.listdir(mask_path):
    # num+=1
    # buffer = mask[mask.find('_') + 1 : ] 
    label = mask[mask.find('_') + 1 : ] 
    if label.startswith('CC'): 
        image = cv2.imread(f'{image_path}/{label}')
        # masks = cv2.imread(f'{mask_path}/{mask}')
        mask_zeros = np.zeros_like(image) 
            
        # if f'{label}.txt' in os.listdir(label_path):
        with open(f'{label_path}/{label}.txt') as f:
            lines = f.readlines()
            data = lines[int(mask[0])] 

            #hands
            # temp1 = data[data.find('|') + 1 : ]
            # temp2 = temp1[temp1.find('|') + 1 : ]
            # temp = temp2[ : temp2.find('|') - 1]

            #objects
            temp1 = data[data.find('|') + 1 : ]
            temp2 = temp1[temp1.find('|') + 1 : ]
            temp3 = temp2[temp2.find('|') + 1 : ]
            temp = temp3[ : temp3.find('|') - 1]

            #tools
            # temp = data[data.rfind('|') + 1 : ] 
            # print(temp) 
            # bbox = temp[temp.rfind('|') + 1 : ].split(",") 


            if temp.strip() != 'None':
                # bbox = temp[temp.rfind('|') + 1 : ].split(",")
                # sys.exit()
                bbox = temp.split(",")
                zero = int(float(bbox[0].strip()))
                one = int(float(bbox[1].strip()))
                two = int(float(bbox[2].strip()))
                three = int(float(bbox[3].strip()))
                mask_read = np.array(Image.open(f'{mask_path}/{mask}'))
                # mask_read = cv2.resize(mask_read, (4000, 4000), interpolation=cv2.INTER_AREA)

                #restoring original mask shape
                width = two - zero
                height = three - one

                x_center = int((mask_read.shape[0]) // 2)
                y_center = int((mask_read.shape[1]) // 2)
                new_zero = x_center - (width  // 2) 
                if width % 2 != 0:
                    new_two = x_center + ((width // 2) + 1)
                else:
                    new_two = x_center + (width // 2)
                new_one = y_center - (height // 2)
                if height % 2 != 0:
                    new_three = y_center + ((height // 2) + 1)
                else:
                    new_three = y_center + (height // 2)

                object_box = mask_read[new_one : new_three, new_zero : new_two] 


                if width > 0 and height > 0:
                    mask_read = cv2.resize(mask_read, (width, height))
                    test = mask_zeros[one : three, zero : two]
                    if width == test.shape[1] or height == test.shape[0]:
                        mask_zeros[one : three, zero : two] = mask_read
                        # cv2.imwrite(f'./test/{mask[:-4]}.png', mask_zeros)  
                        cv2.imwrite(f'{dst_path}/{mask[:-4]}.png', mask_zeros) 

                count+=1
                print(f'Number of in-contact object masks : {count}')
            
