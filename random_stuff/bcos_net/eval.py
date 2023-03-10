from util import *  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet import *


LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 35
NUM_WORKERS = 1
IMAGE_HEIGHT = 368 
IMAGE_WIDTH = 640 
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = '/y/ayhassen/epick_dataset/Images/train' 
TRAIN_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/train/hands_updated'
VAL_IMG_DIR = '/y/ayhassen/epick_dataset/Images/val' 
VAL_MASK_DIR = '/y/ayhassen/epick_dataset/Masks/val/hands_updated'


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

model = Unet().to(DEVICE)

if LOAD_MODEL:
    load_checkpoint(torch.load("./test.pth"), model)

# check_accuracy(train_loader, model, device=DEVICE)  
save_predictions_as_imgs(
            val_loader, model, folder="hands/", device=DEVICE
        ) 