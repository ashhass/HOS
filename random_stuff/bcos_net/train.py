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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# Hyperparameters etc. 
LEARNING_RATE = 1e-5
DECAY_RATE = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 8
IMAGE_HEIGHT = 368  
IMAGE_WIDTH = 640  
BASEWIDTH = 1920
PIN_MEMORY = True
LOAD_MODEL = False 
TRAIN_IMG_DIR = '/y/ayhassen/epick_dataset/Images/train' 
TRAIN_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/train/hands_updated'
VAL_IMG_DIR = '/y/ayhassen/epick_dataset/Images/val' 
VAL_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/val/hands_updated'


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=640, width=640),
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
            A.Resize(height=640, width=640),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = Unet().to(DEVICE)
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
        load_checkpoint(torch.load("./test.pth"), model) 
 
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0
    for epoch in range(NUM_EPOCHS):
        count = 0
        running_loss = 0.0
        loss_values = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # torchvision.utils.save_image(data, './image.jpg')
            # sys.exit()
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
    
        val_loss = validate(model, val_loader, loss_fn)
     

        save_checkpoint(checkpoint) 
        print(f'EPOCH NUMBER :  {epoch}')
        print(f'LEARNING RATE :  {LEARNING_RATE}')
        epoch+=1

writer.flush()

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    val_loss = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            writer.add_scalar("Loss(val) - Epoch", loss)
        total_loss += loss.item()
        val_loss.append(total_loss)
        if batch_idx % 10 == 9:
                print(f'val_loss : {total_loss / 10}')
                total_loss = 0.0
    return val_loss

if __name__ == "__main__":
    main() 
