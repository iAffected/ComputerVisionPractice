import os
import struct

import numpy as np
from PIL import Image

TRAIN_DATASET_RAW_PATH = r'D:\Datasets\MNIST\train-images.idx3-ubyte'
TRAIN_DATASET_DECODE_PATH = r'D:\Datasets\MNIST\train'

data_file = TRAIN_DATASET_RAW_PATH
# It's 47040016B, but we should set to 47040000B
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'
data_buf = open(data_file, 'rb').read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
dataset = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))
dataset = np.array(dataset).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)
label_file = r'D:\Datasets\MNIST\train-labels.idx1-ubyte'
# It's 60008B, but we should set to 60000B
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

dataset_root = TRAIN_DATASET_DECODE_PATH
if not os.path.exists(dataset_root):
    os.mkdir(dataset_root)

for i in range(10):
    file_name = dataset_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for ii in range(numLabels):
    img = Image.fromarray(dataset[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = dataset_root + os.sep + str(label) + os.sep + str(ii) + '.png'
    img.save(file_name)
