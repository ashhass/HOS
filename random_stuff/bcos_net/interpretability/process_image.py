from interpretability.utils import grad_to_img
from interpretability.project_utils import Str2List, to_numpy
from PIL import Image
import numpy as np
import torch
from data_transforms.transforms import AddInverse, MyToTensor
import torchvision
import cv2
import matplotlib.pyplot as plt

@torch.no_grad()
def most_predicted(model, image):
    predictions = []
    for image in image:
        img = torch.from_numpy(image)
        img = img.reshape(img.shape[2], img.shape[1], img.shape[0])
        img = img[None,:,:,:]
        img = torch.tensor(img)
        predictions.append((model((img.float())))[0].argmax().item())
    c_idcs, counts = np.unique(predictions, return_counts=True)
    c_idx = c_idcs[np.argsort(counts)[-1]]

    print("Most predicted class:", c_idx)
    return c_idx

def process_image(model, count, image, class_idx=-1):
    # predictions = []
    atts = []
    imgs = []
    model.eval()
    img = MyToTensor()(Image.fromarray(image))
    img = AddInverse()(img).cuda()[:][None].requires_grad_(True)
    # predictions.append((model(AddInverse()(img)))[0].argmax().item())
    out = model(img)
    out.mean().backward()
    # print(img[:,:,:][0].size())
    att = grad_to_img(img[0], img.grad[0] , alpha_percentile=100, smooth=5)
    att = att.copy(order='C')
    cv2.imwrite(f'./vis/image{count}.jpg', np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8))
    att[..., -1] *= to_numpy(out[0, 0, :,:].sigmoid())
    plt.imsave(f'./vis/vis{count}.jpg', att)
    atts.append(to_numpy(att))
    imgs.append(np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8))


    return imgs, out
