import re
import nltk
import numpy as np
from sets.core import Step


class Tokenize(Step):

    _regex_tag = re.compile(r'<[^>]+>')

    def __call__(self, dataset, columns=None):
        # pylint: disable=arguments-differ
        dataset = dataset.copy()
        columns = columns or dataset.columns
        for column in columns:
            tokens = [list(self._tokenize(x)) for x in dataset[column]]
            padded = self._pad(tokens)
            dataset[column] = padded
        return dataset

    @classmethod
    def _tokenize(cls, sentence):
        """
        Split a sentence while preserving tags.
        """
        while True:
            match = cls._regex_tag.search(sentence)
            if not match:
                yield from cls._split(sentence)
                return
            chunk = sentence[:match.start()]
            yield from cls._split(chunk)
            tag = match.group(0)
            yield tag
            sentence = sentence[(len(chunk) + len(tag)):]

    @staticmethod
    def _split(sentence):
        tokens = nltk.word_tokenize(sentence)
        tokens = [x.lower() for x in tokens]
        return tokens

    @staticmethod
    def _pad(data):
        width = max(len(x) for x in data)
        for index, tokens in enumerate(data):
            missing = width - len(tokens)
            data[index] += ['' for _ in range(missing)]
        return np.array(data)
