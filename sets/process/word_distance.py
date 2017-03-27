import warnings
import numpy as np
from sets.core import Step


class WordDistance(Step):

    def __init__(self, *tags, depth=2):
        self._tags = tags
        self._depth = depth

    def __call__(self, dataset, column):
        # pylint: disable=arguments-differ
        dataset = dataset.copy()
        if 'word_distance' in dataset.columns:
            warnings.warn('override existing column word_distance')
        array_shape = dataset[column].shape[:self._depth]
        data = np.empty(array_shape + (len(self._tags),))
        for index, words in enumerate(dataset[column]):
            positions = self._positions(words)
            data[index] = self._relative_sequence(positions, len(words))
        dataset['word_distance'] = data
        return dataset

    def _positions(self, tokens):
        if not all(x in tokens for x in self._tags):
            raise ValueError('a tag was not found')
        return [np.where(tokens == x)[0][0] for x in self._tags]

    @staticmethod
    def _relative_sequence(positions, length):
        sequence = np.empty((length, len(positions)))
        for index, position in enumerate(positions):
            for current in range(length):
                sequence[current][index] = index - position
        return sequence
