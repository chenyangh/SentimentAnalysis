import numpy as np
from sets.core import Embedding


class OneHot(Embedding):

    def __init__(self, words, depth=1):
        words = np.unique(np.sort(words))
        embeddings = np.eye(len(words))
        super().__init__(words, embeddings, depth)
        assert self.shape == (len(words),)
