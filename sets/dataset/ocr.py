import gzip
import csv
import numpy as np
from sets.core import Step, Dataset


class Ocr(Step):
    """
    Dataset of handwritten words collected by Rob Kassel at the MIT Spoken
    Language Systems Group. Each example contains the normalized letters of the
    word, padded to the maximum word length. Only contains lower case letter,
    capitalized letters were removed.
    From: http://ai.stanford.edu/~btaskar/ocr/
    """

    def __new__(cls, host='http://ai.stanford.edu/~btaskar/ocr/'):
        cls._host = host
        filepath = cls.download('letter.data.gz')
        lines = cls._read(filepath)
        data, target, fold = cls._parse(lines)
        data, target = cls._pad(data, target)
        dataset = Dataset(data=data, target=target, fold=fold)
        return dataset

    @classmethod
    def download(cls, filename):
        # pylint: disable=arguments-differ
        url = cls._host + '/' + filename
        return super().download(url, filename)

    @staticmethod
    def _pad(data, target):
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return data, target

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target, fold = [], [], []
        next_ = None
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
                fold.append(int(line[5]))
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target, fold

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines
