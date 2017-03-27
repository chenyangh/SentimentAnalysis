from sets.core import Step


class Normalize(Step):

    def __init__(self, reference):
        self._means = {}
        self._stds = {}
        self._shapes = {}
        for column in reference.columns:
            data = reference[column]
            shape = data.shape[1:]
            mean, std = data.mean(axis=0), data.std(axis=0)
            assert mean.shape == std.shape == shape
            self._means[column] = mean
            self._stds[column] = std
            self._shapes[column] = shape

    @property
    def columns(self):
        return self._means.columns

    def __call__(self, dataset, columns=None):
        dataset = dataset.copy()
        columns = columns or dataset.columns
        for column in columns:
            self._validate_reference(column)
            self._validate_shape(dataset, column)
        for column in columns:
            data = dataset[column]
            data -= self._means[column]
            data /= self._stds[column]
            dataset[column] = data
        return dataset

    def _validate_reference(self, column):
        if column in self._means:
            return
        message = 'no reference for column {}'
        message = message.format(column)
        raise ValueError(message)

    def _validate_shape(self, dataset, column):
        reference_shape = self._shapes[column]
        dataset_shape = dataset[column].shape[1:]
        if reference_shape == dataset_shape:
            return
        message = 'shape {} differs from {} in reference for column {}'
        message = message.format(dataset_shape, reference_shape, column)
        raise ValueError(message)
