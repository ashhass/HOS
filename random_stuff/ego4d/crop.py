import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

count = 0
list = ['5122adb8-0559-4f92-8077-4a99463ca09d', '28bc1ee7-b0c1-4f30-934a-0ab665779d90',  '1ab9d5f7-0181-458e-a5e7-72ce87501f3e',  '45e463b2-bdd5-407a-ab5c-8a5c5534e078 65403036-df60-4ccb-ac72-2808b841665c', '7dd8f1d3-c197-48cd-81cf-d46d9390920f', '2c27b5f1-4af6-49ad-a43c-3efb0c150868',  '9e9c5b05-c7d4-450d-9850-e6ae83caa9a5',  '2409a5a7-a4ed-4fcb-ad77-024dc20988ca',  '0b6fc89d-bf4b-44f3-82e7-67ee02517459',  '114d86a7-2849-46de-8bb7-8fe1e1a48be8',  '672c902a-7871-4e8f-bae5-6e12e2a4e342', '6052a4ef-1227-4a9a-9883-6344081f4991', '9b8e8dc6-4069-4ef4-b8b2-edeb76894d53',  'e19b1be3-8ee9-44cc-ae7d-99a3591ebba5',  'ac0fa6ee-ce7c-42e4-951c-cea1259ccec3',  '51fc7393-cb83-4178-9ade-6c7853a001e0', '2c78909f-ff5c-4a70-b5ae-2d74e11fbe93', '9df7ece6-df68-4158-a294-06aa242c2b9d', '1b204c9b-30da-47cc-b793-8813169e72c7', '9faf5095-8741-4b2d-8b2e-e803467c7130', 'ba11fcda-0048-4440-a7e8-fd15d1661a27', '31cbf7bb-d464-4ec3-bddd-c387766f8572',  '5c2e910c-84e0-4042-b5d6-880a731c3e67', '002d2729-df71-438d-8396-5895b349e8fd', '719d9e89-4eb2-49ea-be14-dc2637dc303f', 'cf8345a5-30e1-44f9-8751-bd99836b5b1f', '1e64cbac-80af-4aa8-86bd-cc03c081ab1a', '192e4b6b-cbc0-4b6f-995d-fbe11df9044e', '64f466e5-99c9-4ed5-89be-7ef74d0dfdd3', '0fe191ef-c28a-422c-aede-46f8aa8532a6', '597b1ca1-e2ea-4a0e-8be0-31b713151ba2', 'c939aae4-e7c8-453c-8671-40573db0c656', '439a78ee-4776-4b70-9ac2-896f9f22f44c', '7f9f75fd-a660-4635-8890-239c6ad82023', 'def2e8dd-aaf7-467f-aa8f-46f654e6f4e0']
path = '/y/relh/ego4d_data/annotations/fho_scod_train.json'
image_path = '/y/ayhassen/ego4d'


file = open(path)
data = json.load(file)

for elements in data:
    for idx, clips in enumerate(data['clips']):
        if data['clips'][idx]['video_uid'] in list:
            video_uid = data['clips'][idx]['video_uid']
            frame_number = data["clips"][idx]["pre_frame"]["frame_number"]
            or_image = cv2.imread(os.path.join(f"{image_path}/{video_uid}", f'{frame_number}_{video_uid}.jpg'))
            cv2.imwrite('./original_image.jpg', or_image)
            if len(np.unique(or_image) >= 2):
                for index, objects in enumerate(data['clips'][idx]['pre_frame']['bbox']):
                    if data['clips'][idx]['pre_frame']['bbox'][index]['object_type'] != 'left_hand' and data['clips'][idx]['pre_frame']['bbox'][index]['object_type'] != 'right_hand':
                        # print(data['clips'][idx]['pre_frame']['bbox'][index]['object_type'])
                        # print(video_uid)
                        px1 = data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['x']
                        py1 = data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['y']
                        px2 = data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['x'] + data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['width'] 
                        py2 = data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['y'] + data['clips'][idx]['pre_frame']['bbox'][index]['bbox']['height'] 
                        # print(py1, py2, px1, px2)
                        image = or_image[int(py1): int(py2), int(px1) : int(px2)]
                        pad_width = (1920 - image.shape[0]) // 2
                        pad_height = (1440 - image.shape[1]) // 2
                        mask_borderType = cv2.BORDER_CONSTANT
                        image = cv2.copyMakeBorder(image, abs(pad_height), abs(pad_height), abs(pad_width), abs(pad_width), mask_borderType, None, value = (128, 128, 128))
                        # print(data['clips'][idx]['pre_frame']['bbox'][index]['structured_noun'])
                        if len(image) != 0:
                            cv2.imwrite(f'{image_path}/cropped_objects/{frame_number}_{video_uid}_{index}.jpg', image)
                            print(count)
                            count+=1
                    