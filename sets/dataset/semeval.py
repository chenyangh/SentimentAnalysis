import os
import itertools
from zipfile import ZipFile
import re
import requests
from sets.core import Step, Dataset


class SemEvalRelation(Step):
    """
    Task 8 from the SemEval 2010 conference, named 'Multi-Way Classification of
    Semantic Relations Between Pairs of Nominals'. Only the training set is
    returned since we believe targets are not available for the test set.
    From: http://semeval2.fbk.eu/semeval2.php?location=tasks#T11
    """

    DOWNLOAD_PAGE = \
        'http://semeval2.fbk.eu/semeval2.php?' \
        'location=download&task_id=11&datatype=test'
    FILENAME = \
        'SemEval2010_task8_all_data/' \
        'SemEval2010_task8_training/TRAIN_FILE.TXT'

    _regex_line = re.compile(r'^[0-9]+\t"(.*)"$')
    _regex_e1 = re.compile(r'<e1>.*</e1>')
    _regex_e2 = re.compile(r'<e2>.*</e2>')

    def __new__(cls):
        train = cls.disk_cache('train', cls._parse_train)
        return train

    @classmethod
    def _parse_train(cls):
        filepath = cls._download_task()
        with ZipFile(filepath, 'r') as archive:
            with archive.open(cls.FILENAME) as file_:
                return cls._parse(file_)

    @classmethod
    def _download_task(cls):
        filename = 'task8.zip'
        filepath = os.path.join(cls.directory(), filename)
        if os.path.isfile(filepath):
            return filepath
        response = requests.get(cls.DOWNLOAD_PAGE)
        assert response.status_code == 200
        url = re.search(r'get.php?[^"]*', response.text).group(0)
        url = 'http://semeval2.fbk.eu/' + url.replace(' ', '%20')
        return cls.download(url, filename)

    @classmethod
    def _parse(cls, file_):
        paragraphs = itertools.groupby(file_, lambda x: x != b'\r\n')
        paragraphs = [list(g) for k, g in paragraphs if k]
        data = [cls._process_data(x[0]) for x in paragraphs]
        target = [cls._process_target(x[1]) for x in paragraphs]
        return Dataset(data=data, target=target)

    @classmethod
    def _process_data(cls, line):
        line = line.decode('ascii').strip()
        line = cls._regex_line.search(line).group(1)
        line = cls._regex_e1.sub('<e1>', line)
        line = cls._regex_e2.sub('<e2>', line)
        return line

    @staticmethod
    def _process_target(line):
        line = line.decode('ascii').strip()
        return line
