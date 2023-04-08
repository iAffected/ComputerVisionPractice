import os
import struct

import numpy as np
from PIL import Image

TEST_DATASET_RAW_PATH = r'D:\Datasets\MNIST\t10k-images.idx3-ubyte'
TEST_DATASET_DECODE_PATH = r'D:\Datasets\MNIST\test'

data_file = TEST_DATASET_RAW_PATH
# It's 7840016B, but we should set to 7840000B
data_file_size = 7840016
data_file_size = str(data_file_size - 16) + 'B'
data_buf = open(data_file, 'rb').read()
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
dataset = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))
dataset = np.array(dataset).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)
label_file = 'ComputerVisionPractice/Practice2/dataset_raw/t10k-labels.idx1-ubyte'
# It's 10008B, but we should set to 10000B
label_file_size = 10008
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

dataset_root = TEST_DATASET_DECODE_PATH
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
