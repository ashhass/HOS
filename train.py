import sys
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
from models.unet import Unet
from util import (
    load_checkpoint,
    save_checkpoint,
    get_loaders, 
    check_accuracy,
    save_predictions_as_imgs,
) 
import matplotlib.pyplot as plt 
import numpy as np
import torchvision
from models.hr_net import config
from models.hr_net.hrnet import get_seg_model
from losses.focal_loss import FocalLoss
from losses.cross_entropy import CrossEntropy
import timm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# Hyperparameters 
LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 8
IMAGE_HEIGHT = 368  
IMAGE_WIDTH = 640  
PIN_MEMORY = True
LOAD_MODEL = False 

#HANDS
# TRAIN_IMG_DIR = '/y/ayhassen/epick_dataset/Images/train' 
# TRAIN_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/train/hands_updated'
# VAL_IMG_DIR = '/y/ayhassen/epick_dataset/Images/val' 
# VAL_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/val/hands_updated' 

# OBJECTS
TRAIN_IMG_DIR = '/y/ayhassen/epick_object_crop/Images/crop_fit/train' 
TRAIN_MASK_DIR = '/y/ayhassen/epick_object_crop/Masks/crop_fit/train'
VAL_IMG_DIR = '/y/ayhassen/epick_object_crop/Images/crop_fit/val' 
VAL_MASK_DIR = '/y/ayhassen/epick_object_crop/Masks/crop_fit/val' 


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=1000, width=1000),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ), 
            ToTensorV2(),
        ], 
    ) 

    val_transforms = A.Compose(
        [
            A.Resize(height=1000, width=1000),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    config.defrost()
    config.merge_from_file('./models/hr_net/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
    config.merge_from_list(
        [
            'DATASET.NUM_CLASSES', 1, 
            # /home/relh/7.893643061319987_5_model.ckpt
            'MODEL.PRETRAINED', './hrnetv2_w48_imagenet_pretrained.pth', 
        ]
    ) 
    config.freeze()
    model = get_seg_model(config).to(DEVICE)
    # model = Unet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("./checkpoints/hrnet_imagenet_pretrain_object_alldatasets.pth"), model) 
 
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0
    for epoch in range(NUM_EPOCHS):
        count = 0
        running_loss = 0.0
        loss_values = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
            # forward
            with torch.cuda.amp.autocast(): 
                predictions = model(data)
                loss = loss_fn(predictions, targets) 
                writer.add_scalar("Loss(train) - Epoch", loss, epoch) 

            # backward 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer) 
            scaler.update() 
            running_loss += loss.item()
            loss_values.append(running_loss)

            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0

            print(f'NUMBER OF ELEMENTS IN THE TRAIN_LOADER : {count}') 
            count+=1

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        loss = loss_fn(predictions, targets) 
        writer.add_scalar("Loss(train) - Epoch", loss, epoch) 
    
        validate(model, val_loader, loss_fn, epoch)
     

        save_checkpoint(checkpoint) 
        print(f'EPOCH NUMBER :  {epoch}')
        print(f'LEARNING RATE :  {LEARNING_RATE}') 
        epoch+=1

writer.flush()

def validate(model, loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    val_loss = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            writer.add_scalar("Loss(val) - Epoch", loss, epoch)
        total_loss += loss.item()
        val_loss.append(total_loss)
        if batch_idx % 10 == 9:
            print(f'val_loss : {total_loss / 10}')
            total_loss = 0.0
    return val_loss

if __name__ == "__main__":
    main() 