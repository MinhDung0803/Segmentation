import mxnet as mx
from mxnet import image
from gluoncv.data.transforms.presets.segmentation import test_transform
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from gluoncv.data import datasets
from gluoncv.model_zoo.icnet import ICNet
import time
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# using cpu
ctx = mx.cpu(0)


def get_model():
    dataset = 'mhpv1'
    backbone = 'resnet50'
    pretrained_base = True
    all_classes = list(datasets[dataset].CLASSES)
    # print("All classes: ", all_classes[1])
    model = ICNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, ctx=ctx,
                  root='./models/HumanParsing/')
    model.classes = datasets[dataset].CLASSES
    # load model
    model.load_parameters('./runs/mhpv1/icnet/resnet50/epoch_0499_mIoU_0.4210.params', ctx=ctx)
    return model, all_classes


def get_all_images(source_path):
    print("[INFO] - Get all name of images ...")
    list_task = []
    for directories in os.listdir(source_path):
        list_task.append(directories)
    print("[INFO] - Get all name of images - DONE")
    return list_task


def load_input_image_list(source_path_in, list_images_name):
    list_images_cv2 = []
    list_images_read = []
    for i in range(len(list_images_name)):
        image_path_elem = source_path_in + "/" + list_images_name[i]
        img_in = image.imread(image_path_elem)
        img_ori_elem = cv2.imread(image_path_elem)
        list_images_cv2.append(img_ori_elem)
        list_images_read.append(img_in)
    return list_images_read, list_images_cv2


# def load_input_image_list(source_path_in, list_images_name):
#     list_images_cv2 = []
#     list_images_read = []
#     fig_kid = plt.figure(figsize=(30, 2))
#     ax = plt.subplot(1, len(list_images_name)*2, 1)
#     ax.axis('off')
#     j = 0
#     for i in range(len(list_images_name)):
#         j+=2
#         image_path_elem = source_path_in + list_images_name[i]
#         img_in = image.imread(image_path_elem)
#         img_ori_elem = cv2.imread(image_path_elem)
#         list_images_cv2.append(img_ori_elem)
#         list_images_read.append(img_in)
#         ax = plt.subplot(1, len(list_images_name)*2, j)
#         ax.axis('off')
#         img_ori = cv2.cvtColor(img_ori_elem, cv2.COLOR_BGR2RGB)
#         plt.imshow(img_ori)
#         plt.title(str(i))
#     fig_kid.savefig("./outputs/output_testing_images/input.png")
#     return list_images_read, list_images_cv2


def segmentation_process_list(img_list, model, images_list_name):
    mask_list = []
    output_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        print(images_list_name[i], i)
        # segmentation process - measure costing time
        img = test_transform(img, ctx)
        # print('img: ', img.shape)
        start = time.time()
        output = model.predict(img)
        end = time.time()
        print('Prediction time (s): ', end - start)
        # print('output: ', output.shape)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        mask = Image.fromarray(predict.astype('uint8'))
        # append mask and output
        mask_list.append(mask)
        output_list.append(output)
    return mask_list, output_list


def get_part_of_body_list(mask_in_list, output_in_list, img_ori_in_list, num_label, all_images_name, source_path):
    img2_list = []
    # print("mask_in_list: ", len(mask_in_list))
    # print("output_in_list: ", len(output_in_list))
    for i in range(len(mask_in_list)):
        mask_in = mask_in_list[i]
        output_in = output_in_list[i]
        img_ori_in = img_ori_in_list[i]
        # get specific part of body(specific label)
        thresh = np.array(mask_in)
        thresh[thresh == num_label] = 255  # upper clothes (4), pants (6)
        thresh[thresh != 255] = 0
        # apply connected component analysis to the thresholded image
        output_in = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output_in
        areas = stats[:, -1]
        areas[0] = 0  # remove the label-0 (background)
        max_label = areas.argmax()
        img2 = np.zeros(labels.shape)
        img2_list.append(img2)
        img2[labels == max_label] = 255
        h, w = img_ori_in.shape[:2]
        img2 = img2[0:h, 0:w]  # Crop mask corresponding to the original image
        # Apply the mask to the original image
        img_ori_in[img2 == 0] = 255

        # plot out the result
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        path_to_read = source_path + "/" + str(all_images_name[i])
        image_input = cv2.imread(path_to_read)
        image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        plt.imshow(image_input)
        ax.set_title('Input')
        ax = fig.add_subplot(1, 2, 2)
        img_ori_in = cv2.cvtColor(img_ori_in, cv2.COLOR_BGR2RGB)
        plt.imshow(img_ori_in)
        ax.set_title('Result')
        save_path = "./outputs/output_testing_images/" + "result_" + str(all_images_name[i][:-4]) + ".png"
        fig.savefig(save_path)
    return img2_list


# def get_part_of_body_list(mask_in_list, output_in_list, img_ori_in_list, num_label):
#     img2_list = []
#     fig_kid = plt.figure(figsize=(30, 2))
#     ax = plt.subplot(1, len(mask_in_list), 1)
#     ax.axis('off')
#     j = 0
#     # num_label = 4
#     for i in range(len(mask_in_list)):
#         j+=1
#         mask_in = mask_in_list[i]
#         output_in = output_in_list[i]
#         img_ori_in = img_ori_in_list[i]
#         # get specific part of body(specific label)
#         thresh = np.array(mask_in)
#         thresh[thresh == num_label] = 255  # upper clothes (4), pants (6)
#         thresh[thresh != 255] = 0
#         # apply connected component analysis to the thresholded image
#         output_in = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
#         (numLabels, labels, stats, centroids) = output_in
#         areas = stats[:, -1]
#         areas[0] = 0  # remove the label-0 (background)
#         max_label = areas.argmax()
#         # print(max_label)
#         # print(labels.shape)
#         img2 = np.zeros(labels.shape)
#         img2_list.append(img2)
#         img2[labels == max_label] = 255
#         h, w = img_ori_in.shape[:2]
#         img2 = img2[0:h, 0:w]  # Crop mask corresponding to the original image
#         # Apply the mask to the original image
#         img_ori_in[img2 == 0] = 255
#         ax = plt.subplot(1, len(mask_in_list), j)
#         ax.axis('off')
#         img_ori_in = cv2.cvtColor(img_ori_in, cv2.COLOR_BGR2RGB)
#         plt.imshow(img_ori_in)
#         plt.title(str(i))
#         # plt.title(str(list_images_name[i]))
#     fig_kid.savefig("./outputs/output_testing_images/output.png")
#     return img2_list


def get_color_name(R, G, B, color_dataset):
    minimum = 10000
    for i in range(1, len(color_dataset)):
        d_R = abs(R - int(color_dataset.loc[i, "R"]))
        d_G = abs(G - int(color_dataset.loc[i, "G"]))
        d_B = abs(B - int(color_dataset.loc[i, "B"]))
        d = d_R + d_G + d_B
        if d <= minimum:
            minimum = d
            cname = color_dataset.loc[i, "color_name"]
    return cname


def get_color(img2_in, img_ori):
    # get the color of that part
    index = ['R', 'G', 'B', 'color_name']
    color_dataset = pd.read_csv('./models/Color/final_data.csv', names=index)
    len_dataset = len(color_dataset)

    # 1. Get RGB pixels from the region of the masked image
    # reshape the image to be a list of pixels
    im_vec = img_ori.reshape((img_ori.shape[0] * img_ori.shape[1], 3))
    mask2 = img2_in.reshape((img_ori.shape[0] * img_ori.shape[1]))
    indices = np.where(mask2 == 255)[0]
    im_vec = im_vec[indices, :]

    # 2. Cluster the data to extract dominant colors in the selected human part
    # cluster the pixel intensities
    n_clusters = 11
    clt = KMeans(n_clusters)
    clt.fit(im_vec)

    # the cluster centers are our dominant colors.
    COLORS = clt.cluster_centers_
    # convert to integer from float
    COLORS = COLORS.astype(int)

    # Labels
    labels = clt.labels_
    LABELS = list(set(labels))

    # print all the colors in the part of body
    for i in range(len(LABELS)):
        label = LABELS[i]
        n_samples = len(labels[labels == label])
        color = COLORS[i]
        (R, G, B) = color
        c_name = get_color_name(R, G, B, color_dataset)
        print(label, ': ', n_samples, "-", c_name)


if __name__ == '__main__':
    # config
    num_label_in = 4
    source_path = "./samples/testing_segmentation"

    print("[MAIN] - Testing process is starting")

    # get all images name
    all_images_list = get_all_images(source_path)
    # load all images
    list_images_read_out, list_images_cv2_out = load_input_image_list(source_path, all_images_list)
    # load model
    model, all_labels = get_model()
    # segment using model
    mask, output = segmentation_process_list(list_images_read_out, model, all_images_list)
    # get result and save as images
    img2 = get_part_of_body_list(mask, output, list_images_cv2_out, num_label_in, all_images_list, source_path)

    print("[MAIN] - Testing process is DONE")
