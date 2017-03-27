from zipfile import ZipFile
import numpy as np
from sets.core import Embedding


class Glove(Embedding):
    """
    The pretrained word embeddings from the Standford NLP group computed by the
    Glove model. From: http://nlp.stanford.edu/projects/glove/
    """

    URL = 'http://nlp.stanford.edu/data/glove.6B.zip'

    def __init__(self, size=100, depth=1):
        assert size in (50, 100, 300)
        words, embeddings = self.disk_cache('data', self._load, size)
        super().__init__(words, embeddings, depth)
        assert self.shape == (size,)

    @classmethod
    def _load(cls, size):
        filepath = cls.download(cls.URL)
        with ZipFile(filepath, 'r') as archive:
            filename = 'glove.6B.{}d.txt'.format(size)
            with archive.open(filename) as file_:
                return cls._parse(file_)

    @staticmethod
    def _parse(file_):
        words = []
        embeddings = []
        for line in file_:
            chunks = line.split()
            word = chunks[0].decode('utf-8')
            embedding = np.array(chunks[1:]).astype(np.float32)
            words.append(word)
            embeddings.append(embedding)
        return np.array(words), np.array(embeddings)
