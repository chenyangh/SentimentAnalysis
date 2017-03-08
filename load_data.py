import json
from datetime import datetime


def read_data(file_name=''):
    data = []
    print('Reading data start', datetime.now())
    # file_name = 'Electronics_5.json'  # 13 secs loading time
    # file_name = 'Digital_Music_5.json'  # 1 sec loading time

    f = open(file_name, 'r')
    for line in f.readlines():
        tmp = json.loads(line)
        data.append([tmp['reviewText'], tmp['overall']])
    f.close()
    print('Reading data start', datetime.now())
    return data



