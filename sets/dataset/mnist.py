import struct
import array
import gzip
import numpy as np
from sets.core import Step, Dataset


class Mnist(Step):
    """
    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples. It is a
    subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image. It is a good database
    for people who want to try learning techniques and pattern recognition
    methods on real-world data while spending minimal efforts on preprocessing
    and formatting. From: http://yann.lecun.com/exdb/mnist/
    """

    def __new__(cls, host='http://yann.lecun.com/exdb/mnist'):
        cls._host = host
        train = cls.disk_cache('train', cls._train_dataset)
        test = cls.disk_cache('test', cls._test_dataset)
        return train, test

    @classmethod
    def download(cls, url):
        # pylint: disable=arguments-differ
        url = cls._host + '/' + url
        return super().download(url)

    @classmethod
    def _train_dataset(cls):
        data = cls.download('/train-images-idx3-ubyte.gz')
        target = cls.download('/train-labels-idx1-ubyte.gz')
        return cls._read_dataset(data, target)

    @classmethod
    def _test_dataset(cls):
        data = cls.download('/t10k-images-idx3-ubyte.gz')
        target = cls.download('/t10k-labels-idx1-ubyte.gz')
        return cls._read_dataset(data, target)

    @classmethod
    def _read_dataset(cls, data_filename, target_filename):
        data_array, data_size, rows, cols = cls._read_data(data_filename)
        target_array, target_size = cls._read_target(target_filename)
        assert data_size == target_size
        data = np.zeros((data_size, rows, cols))
        target = np.zeros((target_size, 10))
        for i in range(data_size):
            current = data_array[i * rows * cols:(i + 1) * rows * cols]
            data[i] = np.array(current).reshape(rows, cols) / 255
            target[i, target_array[i]] = 1
        return Dataset(data=data, target=target)

    @staticmethod
    def _read_data(filename):
        with gzip.open(filename, 'rb') as file_:
            _, size, rows, cols = struct.unpack('>IIII', file_.read(16))
            target = array.array('B', file_.read())
            assert len(target) == size * rows * cols
            return target, size, rows, cols

    @staticmethod
    def _read_target(filename):
        with gzip.open(filename, 'rb') as file_:
            _, size = struct.unpack('>II', file_.read(8))
            target = array.array('B', file_.read())
            assert len(target) == size
            return target, size
