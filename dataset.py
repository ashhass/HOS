import math
from cv2 import resize
import os
import pdb
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

class CustomDataset(Dataset):

    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(f'{self.image_dir}')   
        self.mask= os.listdir(f'{self.mask_dir}')  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # print(index)
        element = self.images[index]
        # img_path = os.path.join(f'{self.image_dir}', f'{element[:element.rfind("_")]}.jpg')
        img_path = os.path.join(f'{self.image_dir}', element)
        mask_path = os.path.join(f'{self.mask_dir}', element) 
        # img_path = f'{self.image_dir}/train'
        # file = open(f'{self.mask_dir}')
        # data = json.load(file)
        # image = cv2.imread(f'{self.image_dir}/{elements}')
        # # for elements in os.listdir(f'{self.image_dir}/{var}'):
        # for element in data["bbox"]:
        #     if elements in element:
        #         bbox = element[elements]
        #         cropped_image = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        #         mask_borderType = cv2.BORDER_CONSTANT
        #         result1 = image.shape[1] - cropped_image.shape[1]
        #         result2 = image.shape[0] - cropped_image.shape[0]
        #         pad_width = math.ceil(result1 / 2)
        #         pad_height =  math.ceil(result2 / 2)
        #         mask_zeros = np.ones_like(image)
        #         new_image = cv2.copyMakeBorder(cropped_image, abs(pad_height), abs(pad_height), abs(pad_width), abs(pad_width), mask_borderType, None, value = (128, 128, 128))
        #         new_image = cv2.resize(new_image, (image.shape[1], image.shape[0]))
        #         image = np.add(mask_zeros, new_image)

        #--------LOADING------------#
        image = np.array(Image.open(img_path).convert('RGB'))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = np.zeros((4000, 4000))

        # annot_path = "/y/ayhassen/allmerged/AR/labels/"
        # new_image = np.zeros_like(image)
        # for label in os.listdir(annot_path):
        #     if label[label.find('AR'):] == f'{element}.txt': 
        #         with open(f'{annot_path}/{label}') as f:
        #             print('true')
        #             lines = f.readlines()
        #             for count, line in enumerate(lines):
        #                 temp = line[line.rfind('|') + 1 : ]
        #                 if temp.strip() != 'None':
        #                     bbox = temp.split(",")
        #                     zero = int(float(bbox[0].strip()))
        #                     one = int(float(bbox[1].strip()))
        #                     two = int(float(bbox[2].strip()))
        #                     three = int(float(bbox[3].strip()))
        #                     # final_image = image_arr[one : three, zero : two] 

        #                     new_image = image[one: three, zero : two] 
        #                     old_image_height, old_image_width, channels = new_image.shape
        #                     mask_borderType = cv2.BORDER_CONSTANT
        #                     new_image_height = 480
        #                     new_image_width = 640
        #                     color = (128, 128, 128)
        #                     result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
        #                     x_center = (new_image_width - old_image_width) // 2
        #                     y_center = (new_image_height - old_image_height) // 2
        #                     result[y_center:y_center+old_image_height, 
        #                     x_center:x_center+old_image_width] = new_image
                            

        
       
        # sys.exit()
        # new_image = np.zeros(image.shape)
        # for block in data["bbox"]:
        #     if element in block:
            # if f'{element[:element.rfind("_")]}.jpg' in block:
                # point = block[f'{element[:element.rfind("_")]}.jpg']
                # point = block[element]
                # cv2.rectangle(image, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), (0, 255, 0), 4)

                # CROP HERE
                # new_mask = mask[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] 
                # new_image = image[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] 

                # MASK HERE
                # new_mask[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] = mask[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])]    
                # new_image[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] = image[int(point[1]) : int(point[3]), int(point[0]) : int(point[2])] 


        mask = (mask > 128).astype(float)
        # cv2.imwrite('image.jpg', new_image)
        # plt.imsave('mask.jpg', new_mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask, element

