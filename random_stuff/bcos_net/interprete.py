from models.unet import Unet
from interpretability.utils import explanation_mode
from interpretability.process_image import process_image
import cv2
import os
from util import *  


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet().to(DEVICE)
load_checkpoint(torch.load("./hand.pth"), model)
# explanation_mode()

image_path = '/y/ayhassen/epick_dataset/Images/val' 

count = 1
for image in os.listdir(image_path):
    if count <= 10:
        image = cv2.imread(os.path.join(image_path, image))
        image = cv2.resize(image, (640, 640))
        print(image.shape)
        imgs, atts = process_image(model, count, image=image)
        count+=1


# for img in imgs:
    # cv2.imwrite('./image.jpg', img)
# for att in atts:
#     cv2.imwrite('./atts.jpg', att)
# cv2.imwrite('./atts.jpg', atts)


