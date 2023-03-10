import os
import json
import cv2
import shutil
import matplotlib.pyplot as plt
from datetime import timedelta
import cv2
import numpy as np
import os



json_path = "/y/relh/ego4d_data/ego4d.json"
path = '/y/relh/new_ego4d_data/v1/full_scale'

# file = open(json_path)
# data = json.load(file)

# file = open(os.path.join(path, 'annotations/vq_train.json'))
# vq = json.load(file)

# count = 0
# for elements in data:
#     for idx, title in enumerate(vq['videos']):
#         # print(data['videos'][idx]['scenarios'])
#         if len(data['videos'][idx]['scenarios']) != 0:
#             if data['videos'][idx]['scenarios'][0] == "Cooking" and len(data['videos'][idx]['scenarios']) == 1:
#                 for lists in os.listdir(os.path.join(path, 'v1')):
#                     if lists == data['videos'][idx]['video_uid']:
#                         for content in os.listdir(os.path.join(f'{path}/v1', lists)):
#                             shutil.copy2(os.path.join(f'{path}/v1/{lists}', content), '/y/ayhassen/ego4d/Images/train')
#                             print(count)
#                             count+=1 


SAVING_FRAMES_PER_SECOND = 30

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def main(video_file):
    filename, _ = os.path.splitext(video_file)
    filename = filename[filename.rfind('/') + 1:]
    if filename == 'manifest':
        pass
    else:
        # filename += "-opencv"
        # make a folder by the name of the video file

        if not os.path.isdir(f'/y/ayhassen/ego4d/{filename}'):
            os.mkdir(f'/y/ayhassen/ego4d/{filename}')
        
        # read the video file    
        cap = cv2.VideoCapture(video_file)
        # get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # get the list of duration spots to save
        saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
        # start the loop
        count = 0
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, 
                # then save the frame
                frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                cv2.imwrite(os.path.join(f'/y/ayhassen/ego4d/{filename}', f"{count}_{filename}.jpg"), frame) 
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1


if __name__ == "__main__":
    import sys
    for files in os.listdir(path):
        main(os.path.join(path, files)) 