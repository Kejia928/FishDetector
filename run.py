import os
import shutil

import cv2
import torch
import tqdm
from model import initialize_model
import argparse
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset


def make_dir(path):
    """
    Create directory
        :param path: directory path
    """

    # if the path exist, delete it
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return


def video2img(fp_in):
    """
    Cut video into picture
        :param fp_in: input video path
        :return: image path
    """

    # get video file name
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]
    # get input video file absolut path
    fp_in = os.path.abspath(fp_in)

    # create output path
    fp_out = 'video2pic/' + file_name + '/ori_pic'
    fp_out_abs = os.path.abspath(fp_out)
    make_dir(fp_out_abs)

    # open video
    vc = cv2.VideoCapture(fp_in)
    # get frame number
    frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # loop each frame
    desc = "cut " + file_name
    for i in tqdm.tqdm(range(frames), desc=desc):
        # rval is a boolean value that indicates whether the next frame was successfully read from the video.
        # If rval is True, it means that the frame was successfully read, and the frame data is stored.
        # If rval is False, it means that there are no more frames to read and the loop should break.
        rval, frame = vc.read()
        if rval:
            # save the current frame of the video as an image file
            cv2.imwrite(f"""{fp_out_abs}/{str(i).rjust(6, "0")}__{file_name}.png""", frame)
            cv2.waitKey(1)
        else:
            break

    # close the video file
    vc.release()
    print("cut video completed")
    return fp_out


def make_video(img_path):
    """
    Transfer the image to video
        :param img_path: the image folder path
    """

    # set video parameter
    size = (1920, 1080)
    fps = 30
    print("each picture's size is ({},{})".format(size[0], size[1]))
    print("video fps is: " + str(fps))
    file_name = img_path.split("/")[-3]
    make_dir('video2pic/' + file_name + '/yolo_video')
    save_path = 'video2pic/' + file_name + '/yolo_video/' + file_name + '.mp4'
    print("save path: " + save_path)
    img_path = img_path + "/"

    # get the all file name from the input path
    all_files = os.listdir(img_path)
    # sort the all files
    all_files.sort()
    index = len(all_files)
    print("total image:" + str(index))

    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(save_path, fourcc, fps, size)
    img_array = []

    # loop each input image file
    for filename in all_files:
        img = cv2.imread(img_path + filename)
        # if the input image can not be read
        if img is None:
            continue
        # put the image in an array
        img_array.append(img)

    # loop the image array
    desc = "make " + file_name + ".mp4"
    for i in tqdm.tqdm(range(len(img_array)), desc=desc):
        # reset the image size for 1080p video
        img_array[i] = cv2.resize(img_array[i], size)
        # write the image into the video
        videowrite.write(img_array[i])
    print('make video completed')
    return

def detection(fp_in):
    dir_path = '../video2pic/20170929_F_5_1'
    path = dir_path + "/ori_pic"
    save_path = dir_path + "/resnet_output"
    # if os.path.exists(save_path):
    #     os.remove(save_path)
    os.makedirs(save_path, exist_ok=True)
    images = CustomDataset(path)
    images.getImage()
    loader = DataLoader(images, batch_size=1, shuffle=False)

    model.load_state_dict(state_dict)
    model.eval()

    all_file = os.listdir(path)
    all_file.sort()
    for file in tqdm.tqdm(all_file):
        image_path = path + '/' + file
        test_dataset = CustomDataset(image_path)
        test_dataset.getSingleImage()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        result = False
        for image, label in test_loader:
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            result = (preds == label.data)

        img = cv2.imread(image_path)

        have = "Have Fish"
        notHave = "No Fish"
        types = notHave
        if result:
            types = have

        font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 4
        font_color = (0, 0, 0)
        line_color = (0, 0, 255)

        text_size = cv2.getTextSize(types, font, font_scale, 2)[0]
        text_width, text_height = text_size[0], text_size[1]

        cv2.rectangle(img, (0, 0), (text_width + 10, text_height + 10), line_color, -1)
        cv2.putText(img, types, (5, text_height + 5), font, font_scale, font_color, thickness=10)

        # cv2.imshow("Image with label", img)
        # cv2.waitKey(1)
        # save image
        cv2.imwrite(save_path + '/' + file, img)

parser = argparse.ArgumentParser(description='YOLO Detector')
parser.add_argument('--video', type=str, help='Path to the video file')
parser.add_argument('--model', type=str, help='Path to the YOLO model file')
args = parser.parse_args()

video_file = args.video if args.video else "video/SP_N5_20170808_2_4.mp4"
model_file = args.model if args.model else "model/yolov5s_best.pt"

model = initialize_model("resnet18_Dropout", 2, False, False)
state_dict = torch.load('../model/resnet/resnet50_exp8.pt', map_location='cpu')

# image = "../binary_clf_dataset/binary_clf_dataset_test/hasFish/000046__20170929_F_5_3_png.rf.9ef893b0cb042bd1da73738dd173e09f.jpg"

# image = "../binary_clf_dataset/binary_clf_dataset_test/notHasFish/000471__SP_N5_20170808_3_1_png.rf.928a8a70bd186c4ed8c0610882da811e.jpg"

# test_dataset = CustomDataset(image)
# test_dataset.getSingleImage()
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# dir_path = '../video2pic/20170929_F_5_1'
# path = dir_path + "/ori_pic"
# save_path = dir_path + "/resnet_output"
# # if os.path.exists(save_path):
# #     os.remove(save_path)
# os.makedirs(save_path, exist_ok=True)
# images = CustomDataset(path)
# images.getImage()
# loader = DataLoader(images, batch_size=1, shuffle=False)
#
# model.load_state_dict(state_dict)
# model.eval()
#
# all_file = os.listdir(path)
# all_file.sort()
# for file in tqdm.tqdm(all_file):
#     image_path = path + '/' + file
#     test_dataset = CustomDataset(image_path)
#     test_dataset.getSingleImage()
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     result = False
#     for image, label in test_loader:
#         outputs = model(image)
#         _, preds = torch.max(outputs, 1)
#         result = (preds == label.data)
#
#     img = cv2.imread(image_path)
#
#     have = "Have Fish"
#     notHave = "No Fish"
#     types = notHave
#     if result:
#         types = have
#
#     font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_HERSHEY_TRIPLEX
#     font_scale = 4
#     font_color = (0, 0, 0)
#     line_color = (0, 0, 255)
#
#     text_size = cv2.getTextSize(types, font, font_scale, 2)[0]
#     text_width, text_height = text_size[0], text_size[1]
#
#     cv2.rectangle(img, (0, 0), (text_width + 10, text_height + 10), line_color, -1)
#     cv2.putText(img, types, (5, text_height + 5), font, font_scale, font_color, thickness=10)
#
#     # cv2.imshow("Image with label", img)
#     # cv2.waitKey(1)
#     # save image
#     cv2.imwrite(save_path + '/' + file, img)

# set video parameter
size = (1920, 1080)
fps = 30
print("each picture's size is ({},{})".format(size[0], size[1]))
print("video fps is: " + str(fps))
os.makedirs(dir_path + '/resnet_video')
video_path = dir_path + '/resnet_video/' + 'result.mp4'
print("save path: " + video_path)
img_path = save_path + "/"

# get the all file name from the input path
all_files = os.listdir(img_path)
# sort the all files
all_files.sort()
index = len(all_files)
print("total image:" + str(index))

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowrite = cv2.VideoWriter(video_path, fourcc, fps, size)
img_array = []

# loop each input image file
for filename in all_files:
    img = cv2.imread(img_path + filename)
    # if the input image can not be read
    if img is None:
        print(filename + " is error!")
        continue
    # put the image in an array
    img_array.append(img)

# loop the image array
desc = "make mp4"
for i in tqdm.tqdm(range(index), desc=desc):
    # reset the image size for 1080p video
    img_array[i] = cv2.resize(img_array[i], size)
    # write the image into the video
    videowrite.write(img_array[i])
print('make video completed')
