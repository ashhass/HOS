import cv2
import numpy
import sys
from models.unet import Unet
from util import *
import torch

model = Unet().to('cuda')
load_checkpoint(torch.load('./hand.pth'), model)


model.eval()
image = cv2.imread('./image_list/actual_0.png')
res = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
image = torch.from_numpy(res).to('cuda').float()


image = torch.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

print(image.shape)

# torchvision.utils.save_image(image, './test_image.jpg')
data = model(image[None,:,:,:])
print(data)
torchvision.utils.save_image(data, './data.jpg') 