import os
import cv2
import numpy as np

mask_path = '/y/ayhassen/allmerged/fulldata/finalmasks/masks'
label_path = '/y/ayhassen/allmerged/fulldata/mask'
dst_path = '/y/ayhassen/allmerged/fulldata/finalmasks/epick/updated_objects'
num = 1

for mask in os.listdir(mask_path):
    temp = mask[mask.find('_') + 1 : ]
    label = temp[temp.find('_') + 1 : ]
    if temp[temp.find('_') + 1 : ].startswith('EK'):
        if (mask.startswith('3') or mask.startswith('5')):
            # mask_read = cv2.imread(f'{mask_path}/{mask}')
            # mask_final = np.zeros_like(mask_read)
            # with open(f'{label_path}/{label[:-4]}.jpg.txt') as f:
            #     lines = f.readlines()
            #     for count, line in enumerate(lines):
            #         split_frwrd = line[line.find('|') + 1 : ]
            #         split_frwrd2 = split_frwrd[split_frwrd.find('|') + 1 : ]
            #         temp1 = split_frwrd2[split_frwrd2.find('|') + 1: ]
            #         split_bck = temp1[ : temp1.rfind('|') - 1]
            #         temp = split_bck[ : split_bck.rfind('|') - 1]
                    
            #         if temp.strip() != 'None':
            #             bbox = temp.split(",")
            #             zero = int(float(bbox[0].strip()))
            #             one = int(float(bbox[1].strip()))
            #             two = int(float(bbox[2].strip()))
            #             three = int(float(bbox[3].strip()))

            #             mask_final[one : three, zero : two] = mask_read[one : three, zero : two]

            #             cv2.imwrite(f'{dst_path}/{mask}', mask_final) 
            print(num)
            num+=1 