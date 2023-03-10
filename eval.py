from util import *  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.hr_net import config
from models.hr_net.hrnet import get_seg_model
from models.unet import Unet


LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 35
NUM_WORKERS = 1
IMAGE_HEIGHT = 368 
IMAGE_WIDTH = 640 
PIN_MEMORY = True
LOAD_MODEL = True
# TRAIN_IMG_DIR = '/y/ayhassen/epick_dataset/Images/train' 
# TRAIN_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/train/hands_updated'
# # VAL_IMG_DIR = '/y/ayhassen/epick_dataset/Images/val' 
# VAL_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/val/hands_updated' 


TRAIN_IMG_DIR = '/y/ayhassen/epick_object_crop/Images/crop_fit/train' 
TRAIN_MASK_DIR = '/y/ayhassen/epick_object_crop/Masks/crop_fit/train'

VAL_IMG_DIR = '/y/ayhassen/allmerged/fulldata/large_object_crops/image_crops'
VAL_MASK_DIR = '/y/ayhassen/allmerged/AR/AR_objects/mask'


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

config.defrost()
config.merge_from_file('./models/hr_net/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
# /home/relh/7.893643061319987_5_model.ckpt
config.merge_from_list(
    [
        'DATASET.NUM_CLASSES', 1, 
        'MODEL.PRETRAINED', './hrnetv2_w48_imagenet_pretrained.pth', 
    ]
) 
config.freeze()
model = get_seg_model(config).to(DEVICE)
# model = Unet().to(DEVICE)

if LOAD_MODEL:
    load_checkpoint(torch.load("./checkpoints/hrnet_imagenet_pretrain_object_alldatasets.pth"), model)

# check_accuracy(val_loader, model, device=DEVICE) 
save_predictions_as_imgs(
            val_loader, model, folder='/y/ayhassen/allmerged/fulldata/large_object_crops/mask_crops', device=DEVICE
            ) 
