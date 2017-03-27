from sets.core import Step


class Split(Step):

    def __init__(self, *ratios):
        ratios = ratios or 0.66
        ratios = ratios if hasattr(ratios, '__len__') else [ratios]
        ratios = [0] + list(ratios) + [1]
        if list(ratios) != sorted(ratios):
            raise ValueError('ratios must be in order')
        if len(ratios) != len(set(ratios)):
            raise ValueError('ratios must be unique')
        self._ratios = ratios

    def __call__(self, dataset):
        splits = [int(len(dataset) * x) for x in self._ratios]
        for start, end in zip(splits[:-1], splits[1:]):
            yield dataset[start:end]
