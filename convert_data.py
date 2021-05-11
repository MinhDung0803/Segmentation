import cv2
import numpy as np
from glob import glob
import os
import shutil
from sklearn.model_selection import train_test_split


def convert_file_label_to_camvid_data_training():
    label_file = "label_colors.txt"
    with open(label_file, "r+") as fr:
        lines = fr.readlines()
        rgb_list = []
        label_list = []
        for line in lines:
            line = line.rstrip()
            rgb_list.append([int(line.split(' ')[2]), int(line.split(' ')[1]), int(line.split(' ')[0])])
            label_list.append(' '.join(line.split(' ')[3:]))
    return rgb_list


def check_label_after_convert():
    path = "test.png"
    image = cv2.imread(path)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            print(image[row, col])


def rename_images_and_annotation(task_num_in):
    images_path = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/default/"
    images_path_output = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/default_output/"
    if not os.path.isdir(images_path_output):
        os.mkdir(images_path_output)
    # rename images
    all_images_list = []
    for directories in os.listdir(images_path):
        all_images_list.append(directories)

    for i in range(len(all_images_list)):
        one_images_path = images_path + all_images_list[i]
        # read image
        image = cv2.imread(one_images_path)
        if task_num_in not in all_images_list[i]:
            output_image_name = images_path_output + task_num_in + "_" + all_images_list[i]
        else:
            output_image_name = images_path_output + all_images_list[i]
        # print(output_image_name)
        # write images
        cv2.imwrite(output_image_name, image)
        convert_PNG_to_JPG(output_image_name)

    annotations_path = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/defaultannot/"
    annotations_path_output = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/defaultannot_output/"
    if not os.path.isdir(annotations_path_output):
        os.mkdir(annotations_path_output)
    # rename annotations
    all_annotations_list = []
    for directories in os.listdir(annotations_path):
        all_annotations_list.append(directories)

    for i in range(len(all_annotations_list)):
        one_annotations_path = annotations_path + all_annotations_list[i]
        # read image
        image = cv2.imread(one_annotations_path)
        if task_num_in not in all_annotations_list[i]:
            output_annotation_name = annotations_path_output + task_num_in + "_" + all_annotations_list[i]
        else:
            output_annotation_name = annotations_path_output + all_annotations_list[i]
        # print(output_annotation_name)
        # write images
        cv2.imwrite(output_annotation_name, image)


def convert_label_using_numpy_array(task_num_in):
    # This function is using to convert data format from CamVid format
    path = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/defaultannot_output/"
    path_save = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_in) + "/result/"
    rgb_list = convert_file_label_to_camvid_data_training()
    list_image = glob(path + "*")
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx, image_path in enumerate(list_image):
        print("{}/{}".format(idx, len(list_image)), "processing image: ", image_path)
        image = cv2.imread(image_path)
        for label in rgb_list:
            index_row, index_col = np.where((image[:, :, 0] == label[0]) & (image[:, :, 1] == label[1]) & (image[:, :, 2] == label[2]))
            for i in range(len(index_col)):
                image[index_row[i], index_col[i]] = rgb_list.index(label)
        cv2.imwrite(path_save + image_path.split("/")[-1], image)


# def main():
#     path = "dataset2/annotations_prepped_train/"
#     path_save = "dataset2/annotations_prepped_train_convert/"
#     rgb_list = convert_file_label_to_camvid_data_training()
#
#     list_image = glob(path + "*")
#     if not os.path.exists(path_save):
#         os.makedirs(path_save)
#
#     for idx, image_path in enumerate(list_image):
#         print("{}/{}".format(idx, len(list_image)), "processing image: ", image_path)
#         image = cv2.imread(image_path)
#         for row in range(image.shape[0]):
#             for col in range(image.shape[1]):
#                 for label in rgb_list:
#                     if label == list(image[row, col]):
#                         image[row, col] = rgb_list.index(label)
#                         break
#         cv2.imwrite(path_save + image_path.split("/")[-1], image)


def copytree_annotations(task_num_input, symlinks=False, ignore=None):
    # if not os.path.isdir("/home/gg-greenlab/Downloads/data_body_segmentaion/data/task04/annotations/"):
    #     os.mkdir("/home/gg-greenlab/Downloads/data_body_segmentaion/data/task04/annotations/")
    src = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_input) + "/result/"
    # main data location
    dst = "/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1/annotations/"
    # # for test
    # dst = "/home/gg-greenlab/Downloads/data_body_segmentaion/test_data/annotations"
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def copytree_images(task_num_input_im, symlinks=False, ignore=None):
    # if not os.path.isdir("/home/gg-greenlab/Downloads/data_body_segmentaion/data/task04/images/"):
    #     os.mkdir("/home/gg-greenlab/Downloads/data_body_segmentaion/data/task04/images/")
    src = "/home/gg-greenlab/Downloads/data_body_segmentaion/data/" + str(task_num_input_im) + "/default_output/"
    # main data location
    dst = "/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1/images/"
    # # for test
    # dst = "/home/gg-greenlab/Downloads/data_body_segmentaion/test_data/images"
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def get_all_task():
    list_task = []
    for directories in os.listdir('/home/gg-greenlab/Downloads/data_body_segmentaion/data'):
        list_task.append(directories)
    # print("list of all tasks: ", list_task)
    return list_task


def split_train_test_data(test_size_value):
    list_images = []
    # main data location
    data_path = "/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1/images"
    # # for test
    # data_path = "/home/gg-greenlab/Downloads/data_body_segmentaion/test_data/images"
    for directories in os.listdir(data_path):
        list_images.append(directories)
    # split data
    X_train, X_test = train_test_split(list_images, test_size=test_size_value, random_state=1)
    X_train = sorted(X_train)
    X_test = sorted(X_test)
    # for train_list file
    # main data location
    textfile_train = open("/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1/train_list.txt", "w")
    # # for test
    # textfile_train = open("/home/gg-greenlab/Downloads/data_body_segmentaion/test_data/train_list.txt", "w")
    for element in X_train:
        textfile_train.write(element + "\n")
    textfile_train.close()
    # for test_list file
    # main data location
    textfile_test = open("/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1/test_list.txt", "w")
    # # for test
    # textfile_test = open("/home/gg-greenlab/Downloads/data_body_segmentaion/test_data/test_list.txt", "w")
    for element in X_test:
        textfile_test.write(element + "\n")
    textfile_test.close()


def convert_PNG_to_JPG(png_path):
    img = cv2.imread(png_path)
    jpg_path = png_path.replace('.png','.jpg')
    cv2.imwrite(jpg_path, img)
    os.remove(png_path)


def convert_PNGdir_to_JPGdir(png_dir='PNG_images'):
    png_paths = sorted(glob.glob(os.path.join(png_dir, '*.png')))
    for png_path in png_paths:
        convert_PNG_to_JPG(png_path)


if __name__ == '__main__':
    # get all task to convert
    all_tasks = get_all_task()

    # check name and rename of default and defaultannot
    print("Running - Rename process...")
    for i in range(len(all_tasks)):
        task_num_ele_rename = all_tasks[i]
        rename_images_and_annotation(task_num_ele_rename)

    # convert data, copy to main data location, split data
    print("Running - Convert data format process...")
    for i in range(len(all_tasks)):
        task_num_ele = all_tasks[i]
        # print("task_num_ele: ", task_num_ele)
        # convert data
        convert_label_using_numpy_array(task_num_ele)
        # copy annotations data to main data location for training
        copytree_annotations(task_num_ele)
        # copy images data to main data location for training
        copytree_images(task_num_ele)
    # split datda for training and testing task
    test_size_value_input = 0.2
    split_train_test_data(test_size_value_input)
    print("All processes done")
