import cv2
import os
from multiprocessing import Pool


def resize(frame_path):
    frame_path_s = frame_path[0]
    frame_folder_t = frame_path[1]
    frame = frame_path[2]
    num = frame.split("_")[-1].split(".")[0]
    frame_path_t = frame_folder_t + "0" + num + ".png"

    print(frame_path_t)

    img = cv2.imread(frame_path_s)

    height = int(img.shape[0])
    width = int(img.shape[1])

    if height < width:
        center = int(width / 2)
        crop = img[:, center - int(height/2) : center + int(height/2)]
    else:
        center = int(height / 2)
        crop = img[center - int(width/2) : center + int(width/2) , :]

    # print(height, width, center)
    # print(crop.shape[0], crop.shape[1])

    resized = cv2.resize(crop, (224, 224))
    cv2.imwrite(frame_path_t, resized)


folder_path_s = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames/"
folder_path_t = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/"

# for sub_folder_s_name in os.listdir(folder_path_s):
#
#     sub_folder_s = folder_path_s + sub_folder_s_name + "/"
#     sub_folder_t = folder_path_t + sub_folder_s_name + "/"
#     os.mkdir(sub_folder_t)
#
#     for charactor_folder_s_name in os.listdir(sub_folder_s):
#         charactor_folder_s = sub_folder_s + charactor_folder_s_name + "/"
#         charactor_folder_t = sub_folder_t + charactor_folder_s_name + "/"
#         os.mkdir(charactor_folder_t)
#
#         for frame_folder_s_name in os.listdir(charactor_folder_s):
#             frame_folder_s = charactor_folder_s + frame_folder_s_name + "/"
#             frame_folder_t = charactor_folder_t + frame_folder_s_name + "/"
#             os.mkdir(frame_folder_t)
#
#
#             # single processing
#             # for frame in os.listdir(frame_folder_s):
#
#             #     frame_path_s = frame_folder_s + frame
#             #     frame_path_t = frame_folder_t + frame
#             #     #print(frame_path_s)
#
#             #     img = cv2.imread(frame_path_s)
#
#             #     height = int(img.shape[0])
#             #     width = int(img.shape[1])
#
#             #     if height < width:
#             #         center = int(width / 2)
#             #         crop = img[:, center - int(height/2) : center + int(height/2)]
#             #     else:
#             #         center = int(height / 2)
#             #         crop = img[center - int(width/2) : center + int(width/2) , :]
#
#             #     # print(height, width, center)
#             #     # print(crop.shape[0], crop.shape[1])
#
#             #     resized = cv2.resize(crop, (224, 224))
#             #     cv2.imwrite(frame_path_t, resized)
#
#
#             # multi processing
#             frame_path = []
#             for frame in os.listdir(frame_folder_s):
#                 frame_path_s = frame_folder_s + frame
#                 frame_path_t = frame_folder_t + frame
#
#                 frame_path.append([frame_path_s, frame_folder_t, frame])
#
#             #print(frame_path)
#             with Pool() as p:
#                 p.map(resize, frame_path)


sub_folder_s = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames/Style_3/"
sub_folder_t = "/home/snamburu/Storage/stylizeddata/SyntheticV1-stylized-ReReVST-frames_resized/Style_3/"


for charactor_folder_s_name in os.listdir(sub_folder_s):
    charactor_folder_s = sub_folder_s + charactor_folder_s_name + "/"
    charactor_folder_t = sub_folder_t + charactor_folder_s_name + "/"
    os.mkdir(charactor_folder_t)

    for frame_folder_s_name in os.listdir(charactor_folder_s):
        frame_folder_s = charactor_folder_s + frame_folder_s_name + "/"
        frame_folder_t = charactor_folder_t + frame_folder_s_name + "/"
        os.mkdir(frame_folder_t)


        # single processing
        # for frame in os.listdir(frame_folder_s):

        #     frame_path_s = frame_folder_s + frame
        #     frame_path_t = frame_folder_t + frame
        #     #print(frame_path_s)

        #     img = cv2.imread(frame_path_s)

        #     height = int(img.shape[0])
        #     width = int(img.shape[1])

        #     if height < width:
        #         center = int(width / 2)
        #         crop = img[:, center - int(height/2) : center + int(height/2)]
        #     else:
        #         center = int(height / 2)
        #         crop = img[center - int(width/2) : center + int(width/2) , :]

        #     # print(height, width, center)
        #     # print(crop.shape[0], crop.shape[1])

        #     resized = cv2.resize(crop, (224, 224))
        #     cv2.imwrite(frame_path_t, resized)


        # multi processing
        frame_path = []
        for frame in os.listdir(frame_folder_s):
            frame_path_s = frame_folder_s + frame
            frame_path_t = frame_folder_t + frame

            frame_path.append([frame_path_s, frame_folder_t, frame])

        #print(frame_path)
        with Pool() as p:
            p.map(resize, frame_path)
