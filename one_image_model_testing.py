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
    model.load_parameters('./models/HumanParsing/icnet_resnet50_mhpv1.params', ctx=ctx)
    return model, all_classes


def load_input_image(image_path):
    img_in = image.imread(image_path)
    img_ori_in = cv2.imread(image_path)
    cv2.imshow("input image", img_ori_in)
    print("Press any key to continue the process")
    cv2.waitKey()
    return img_in, img_ori_in


def segmentation_process(img, model):
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
    print(len(output))
    # pixels = list(mask.getdata())
    return mask, output


def get_part_of_body(mask_in, output_in):
    # get specific part of body(specific label)
    thresh = np.array(mask_in)
    num_label = int(input("Please select the number of label - upper clothes (4), pants (6): "))
    thresh[thresh == num_label] = 255  # upper clothes (4), pants (6)
    thresh[thresh != 255] = 0
    # apply connected component analysis to the thresholded image
    output_in = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output_in
    areas = stats[:, -1]
    areas[0] = 0  # remove the label-0 (background)
    max_label = areas.argmax()
    # print(max_label)
    # print(labels.shape)
    img2 = np.zeros(labels.shape)
    img2[labels == max_label] = 255
    h, w = img_ori.shape[:2]
    img2 = img2[0:h, 0:w]  # Crop mask corresponding to the original image
    # Apply the mask to the original image
    img_ori[img2 == 0] = 255
    # img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    cv2.imshow("output image", img_ori)
    return img2


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


def get_color(img2_in):
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
    model, all_labels = get_model()
    # Load and visualize the image
    filename = './samples/images/task04_person_600_11.jpg'
    img, img_ori = load_input_image(filename)
    mask, output = segmentation_process(img, model)
    img2 = get_part_of_body(mask, output)
    get_color(img2)
    print("all done")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
