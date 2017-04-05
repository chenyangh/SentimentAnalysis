"""Functions for reading MNIST data"""

import gzip
import os
import numpy as np


def read_revfloat32(bytestream):
    """Read a float32 in reverse byte order"""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]"""
    print('Extracting', filename)
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = read_revfloat32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                 (magic, filename))
            num_images = read_revfloat32(bytestream)
            rows = read_revfloat32(bytestream)
            cols = read_revfloat32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index]"""
    print('Extracting', filename)
    with open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = read_revfloat32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, filename))
            num_items = read_revfloat32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return dense_to_one_hot(labels, num_classes)
            return labels


def read_data(datapath, image_shape=[784], image_range=[0.0, 1.0],
              one_hot=False, num_validation=0, permute_pixels=False):
    """Read MNIST images and labels"""
    # input image filenames
    if permute_pixels:
        train_image_filename = os.path.join(datapath,
                                            "train-perm-images-idx3-ubyte.gz")
        test_image_filename = os.path.join(datapath,
                                           "t10k-perm-images-idx3-ubyte.gz")
    else:
        train_image_filename = os.path.join(datapath, "train-images-idx3-ubyte.gz")
        test_image_filename = os.path.join(datapath, "t10k-images-idx3-ubyte.gz")
        # output label filenames
    train_label_filename = os.path.join(datapath, "train-labels-idx1-ubyte.gz")
    test_label_filename = os.path.join(datapath, "t10k-labels-idx1-ubyte.gz")
    # train/validation inputs
    if np.prod(image_shape) != 784 or len(image_shape) > 2:
        raise ValueError("Invalid shape specified for MNIST images")
    if len(image_range) != 2 or image_range[0] >= image_range[1]:
        raise ValueError("Invalid range specified for MNIST images")
    Xin_img = extract_images(train_image_filename).astype(np.float32)
    if len(image_shape) == 1:
        Xin = Xin_img.reshape([Xin_img.shape[0]] + image_shape)
    elif len(image_shape) == 2:
        Xin = Xin_img.reshape([Xin_img.shape[0]] + image_shape + [1])
    Xin *= (image_range[1] - image_range[0]) / np.max(Xin)
    Xin += image_range[0]
    # train/validation outputs
    if one_hot:
        Yin = extract_labels(train_label_filename, one_hot=True).astype(np.float32)
    else:
        Yin = extract_labels(train_label_filename, one_hot=False).astype(np.int64)
        # split into separate training and validation sets
    Xva = Xin[:num_validation, ...]
    Yva = Yin[:num_validation, ...]
    Xtr = Xin[num_validation:, ...]
    Ytr = Yin[num_validation:, ...]
    # test inputs
    Xte_img = extract_images(test_image_filename).astype(np.float32)
    if len(image_shape) == 1:
        Xte = Xte_img.reshape([Xte_img.shape[0]] + image_shape)
    elif len(image_shape) == 2:
        Xte = Xte_img.reshape([Xte_img.shape[0]] + image_shape + [1])
    Xte *= (image_range[1] - image_range[0]) / np.max(Xte)
    Xte += image_range[0]
    # test outputs
    if one_hot:
        Yte = extract_labels(test_label_filename, one_hot=True).astype(np.float32)
    else:
        Yte = extract_labels(test_label_filename, one_hot=False).astype(np.int64)
        # return train, validation, test pairs
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)
