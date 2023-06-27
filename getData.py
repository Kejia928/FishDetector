import os
import shutil

if not os.path.exists("../binary_clf_dataset"):
    os.mkdir("../binary_clf_dataset")
if not os.path.exists("../binary_clf_dataset/binary_clf_dataset_train"):
    os.mkdir("../binary_clf_dataset/binary_clf_dataset_train")
if not os.path.exists("../binary_clf_dataset/binary_clf_dataset_train/hasFish"):
    os.mkdir("../binary_clf_dataset/binary_clf_dataset_train/hasFish")
if not os.path.exists("../binary_clf_dataset/binary_clf_dataset_train/notHasFish"):
    os.mkdir("../binary_clf_dataset/binary_clf_dataset_train/notHasFish")

original_path = "../image_dataset/coco128/images/train/"
all_files = os.listdir("../image_dataset/coco128/images/train/")

for file in all_files:
    filename = file.split(".")
    print(filename)
    print(filename.pop())
    print(filename)
    txt_filename = ".".join(filename)
    txt_filename = txt_filename + ".txt"
    print(txt_filename)
    path = "image_dataset/coco128/labels/train/" + txt_filename
    with open(path, 'r') as f:
        for line in f:
            line = line.split(" ")
            if line[0] == "1":
                shutil.copy(original_path + file, "../binary_clf_dataset/binary_clf_dataset_train/hasFish")
                break

hasHish = os.listdir("../binary_clf_dataset/binary_clf_dataset_train/hasFish")
for file in all_files:
    if file not in hasHish:
        shutil.copy(original_path + file, "../binary_clf_dataset/binary_clf_dataset_train/notHasFish")
