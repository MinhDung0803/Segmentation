import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.cpu(0)

# Prepare the image
url = 'https://github.com/zhanghang1989/image-data/blob/master/encoding/' + \
    'segmentation/ade20k/ADE_val_00001755.jpg?raw=true'
filename = 'ade20k_example.jpg'
gluoncv.utils.download(url, filename, True)

# Load the image
img = image.imread(filename)

from matplotlib import pyplot as plt
plt.imshow(img.asnumpy())
plt.show()

# normalize the image using dataset mean
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

# Load the pre-trained model and make prediction
model_names = ['deeplab_resnet101_coco', 'deeplab_resnet101_ade', 'deeplab_resnet101_citys']
model_name = model_names[0]

model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
### Downloading /root/.mxnet/models/deeplab_resnet101_ade-bf1584df.zip from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/deeplab_resnet101_ade-bf1584df.zip.

# make prediction using single scale
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

# Add color pallete for visualization
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'ade20k')
mask.save('output.png')

# show the predicted mask
mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()



