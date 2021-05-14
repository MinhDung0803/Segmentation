import cv2
import numpy as np
from glob import glob
import os
import shutil
from sklearn.model_selection import train_test_split


def get_all_task(input_path):
    print("[INFO] - Get all name of images ...")
    list_task = []
    for directories in os.listdir(input_path):
        list_task.append(directories)
    print("[INFO] - Get all name of images - DONE")
    return list_task


def rename(main_path, list_images, image_name, destination_path):
    print("[INFO] - Renaming all images and copy to new location ...")
    for i in range(len(list_images)):
        image_path = main_path + "/" + list_images[i]
        img = cv2.imread(image_path)
        des_path = destination_path + "/" + str(image_name) + "_" + str(i) + ".jpg"
        cv2.imwrite(des_path, img)
    print("[INFO] - Renaming all images and copy to new location - DONE")


if __name__ == '__main__':
    # path for easy testing picture
    path_input = "/home/gg-greenlab/Pictures/test_segmentation"
    # get all task to convert
    all_images = get_all_task(path_input)
    image_name_in = "e"
    destination_path_input = "/storages/data/dungpm/code/Segmentation/samples/testing_segmentation"
    # rename for easy testing picture
    rename(path_input, all_images, image_name_in, destination_path_input)

    # rename for hard testing picture
    path_input_h = "/home/gg-greenlab/Pictures/test_segmentation_hard"
    # get all task to convert
    all_images_h = get_all_task(path_input_h)
    image_name_in_h = "h"
    # rename for hard testing picture
    rename(path_input_h, all_images_h, image_name_in_h, destination_path_input)
