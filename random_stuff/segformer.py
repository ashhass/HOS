import os
import numpy as np
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig, SegformerModel
import cv2
import torchvision
import torch.nn as nn


configuration = SegformerConfig()
model = SegformerModel(configuration)
configuration = model.config


num_classes = 2

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            return_dict=False, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

image = cv2.imread('./test.jpg')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits 

upsampled_logits = nn.functional.interpolate(
            logits, 
            size=(128, 128), 
            mode="bilinear", 
            align_corners=False
        )

print(logits.shape)
predicted = upsampled_logits.argmax(dim=1)
predicted = predicted.detach().cpu().numpy()
predicted = predicted.reshape(128, 128, 1)
print(predicted.shape)
print(np.unique(predicted))

cv2.imwrite('./test1.jpg', predicted)
# torchvision.utils.save_image(predicted, './image.jpg') 
