import numpy as np
from sets.core import Step


class Concat(Step):

    def __init__(self, axis, target='data'):
        if axis < 1:
            raise ValueError('concat axis must be one or higher')
        self._axis = axis
        self._target = target

    def __call__(self, dataset, columns=None):
        dataset = dataset.copy()
        columns = columns or dataset.columns
        arrays = [dataset[x] for x in columns]
        result = np.concatenate(arrays, axis=self._axis)
        del dataset[columns]
        dataset[self._target] = result
        return dataset
