import random
import numpy as np


class Dataset:
    """
    A mapping from column names to immutable arrays of equal length.
    """

    def __init__(self, **data):
        self._data = {}
        self._length = None
        super().__init__()
        for column, data in data.items():
            self[column] = data

    @property
    def columns(self):
        return sorted(self._data.keys())

    def copy(self):
        data = {x: self[x].copy() for x in self.columns}
        return type(self)(**data)

    def sample(self, size):
        indices = random.sample(range(len(self)), size)
        return self[indices]

    def __len__(self):
        return self._length

    def __contains__(self, column):
        return column in self._data

    def __getattr__(self, column):
        if column in self:
            return self[column]
        raise AttributeError

    def __iter__(self):
        for index in range(len(self)):
            yield tuple(self[x][index] for x in self.columns)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.columns != other.columns:
            return False
        for column in self.columns:
            if not (self[column] == other[column]).all():
                return False
        return True

    def __getitem__(self, key):
        if isinstance(key, slice):
            data = {x: self[x][key] for x in self.columns}
            return type(self)(**data)
        if isinstance(key, (tuple, list)) and isinstance(key[0], int):
            data = {x: self[x][key] for x in self.columns}
            return type(self)(**data)
        if isinstance(key, (tuple, list)) and isinstance(key[0], str):
            data = {x: self[x] for x in key}
            return type(self)(**data)
        return self._data[key].copy()

    def __setitem__(self, key, data):
        if isinstance(key, (tuple, list)) and isinstance(key[0], str):
            for column, data in zip(key, data):
                self[column] = data
            return
        if isinstance(key, (tuple, list)) and isinstance(key[0], int):
            raise NotImplementedError('column content is immutable')
        data = np.array(data)
        data.setflags(write=False)
        if not data.size:
            raise ValueError('must not be empty')
        if not self._length:
            self._length = len(data)
        if len(data) != self._length:
            raise ValueError('must have same length')
        self._data[key] = data

    def __delitem__(self, key):
        if isinstance(key, (tuple, list)):
            for column in key:
                del self._data[column]
            return
        del self._data[key]

    def __str__(self):
        message = ''
        for column in self.columns:
            message += '{} ({}):\n\n'.format(column, self[column].dtype)
            message += str(self[column]) + '\n\n'
        return message

    def __getstate__(self):
        return {'length': self._length, 'data': self._data}

    def __setstate__(self, state):
        self._length = state['length']
        self._data = state['data']
