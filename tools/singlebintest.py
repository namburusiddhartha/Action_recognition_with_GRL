
import cv2
import numpy as np
import struct
import argparse
import glob
import os
import tqdm
import tensorflow as tf
import json
import random
import sys
import xml.etree.ElementTree as ET

gesture_names = ['Advance', 'Attention', 'Rally', 'MoveForward', 'Halt', 'FollowMe', 'MoveInReverse']
models = ['FemaleMilitary', 'FemaleCivilian', 'MaleMilitary', 'MaleCivilian']

xmlfolderpath = "/home/snamburu/Storage/Shared Data/Gestures_Celso/SyntheticV1/"

folderpath = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames"

['/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/Style_7/MaleCivilian/18920_Halt_13_1_2019_17_36_58-no-global/label.bin',
'/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/Style_2/MaleMilitary/17210_MoveForward_13_1_2019_22_3_17-no-global/label.bin',
'/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/Style_6/MaleCivilian/14858_MoveForward_13_1_2019_10_43_26/label.bin']



styles = os.listdir(folderpath)

fps_multiplier = 1.0

# for x in styles:
#     stylenum  = os.path.join(folderpath,x)
#     for y in os.listdir(stylenum):
#         inpath = os.path.join(stylenum,y)
#         for z in os.listdir(inpath):
inpath = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/Style_6/MaleCivilian"
videopath = "MaleCivilian/14858_MoveForward_13_1_2019_10_43_26"#os.path.join(y,z)[:-10]
xmlpath = os.path.join(xmlfolderpath, videopath)
xml_filename = os.path.join(xmlpath + '.xml')
video_path = os.path.join(xmlpath + '.mp4')
print(xml_filename)
xml_str = open(xml_filename, 'r').read().encode('utf-16-be')
tree = ET.fromstring(xml_str)
#except:

gesture = tree.find('gesture').text
gesture_idx = gesture_names.index(gesture)
frame_gesture_idx = gesture_idx + 1
print(frame_gesture_idx)
start_time = float(tree.find('startTime').text)
end_time = float(tree.find('endTime').text)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(frame_count)
start_frame = int(fps * start_time)
end_frame = int(fps * end_time)
start_frame = int(fps_multiplier * fps * start_time)
end_frame = int(fps_multiplier * fps * end_time)

z = "14858_MoveForward_13_1_2019_10_43_26-no-global/"
frames_dir = os.path.join(inpath,z)
label_output_path = os.path.join(frames_dir, 'label.bin')
print(label_output_path)
with open(label_output_path, 'wb') as f_c:
    for i in range(int(frame_count)):
        frame_gesture_idx2 = frame_gesture_idx
        if i < start_frame or i > end_frame:
            frame_gesture_idx2 = 0
            repetitions_to_save = 0
        f_c.write(struct.pack('<i', int(frame_gesture_idx2)))


#TEST IF THE label.bin IS WORKING OR NOT
frame_count_f = len(glob.glob(os.path.join(frames_dir, '*[0-9].png')))
#frame_count_f = len(glob.glob(os.path.join(frames_dir, '*[0-9].jpeg')))
fmt_str = '<{}i'.format(134)
labels = list(struct.unpack(fmt_str, open(os.path.join(frames_dir, 'label.bin'), 'rb').read()))
print(labels)
print(len(labels))
print(frame_count_f)
