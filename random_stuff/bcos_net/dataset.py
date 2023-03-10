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

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir 
        self.transform = transform
        self.images = os.listdir(self.image_dir)   
        self.mask = os.listdir(self.mask_dir)   

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        element = self.mask[index]
        img_path = os.path.join(self.image_dir, f'{element[:element.rfind("_")]}.jpg')
        # img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.mask[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 128).astype(float)
            
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]


        return image, mask  