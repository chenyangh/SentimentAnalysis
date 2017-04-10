import random
import numpy as np
import re


def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def p1():
    file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
    select_ratio = 0.9
    split_point = np.int32(5331 * 0.9)
    for file in file_list:
        with open('data/' + file, 'r', encoding="ISO-8859-1") as f1:
            lines = f1.readlines()
            random.shuffle(lines)
            lines_train = lines[: split_point]
            lines_test = lines[split_point:]
            with open('data/' + file + '.train', 'w') as f2:
                for line in lines_train:
                    line = clean_str(line.strip())
                    f2.write(line + '\r')
            with open('data/' + file + '.test', 'w') as f2:
                for line in lines_test:
                    line = clean_str(line.strip())
                    f2.write(line + '\r')

    with open('vocb.txt', 'w') as wf:
        voc_dict = {}
        for file in file_list:
            with open('data/' + file + '.train', 'r', encoding="ISO-8859-1") as rf:
                for word in rf.read().split():
                    if word not in voc_dict:
                        voc_dict[word] = 1
                    else:
                        voc_dict[word] += 1
        idx = 0
        for voc in voc_dict:
            wf.write(voc + ' ' + str(idx) + '\n')
            idx += 1


def p1_1():
    file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
    select_ratio = 0.9
    split_point = np.int32(5331 * 0.9)
    for file in file_list:
        with open('data/' + file, 'r', encoding="ISO-8859-1") as f1:
            lines = f1.readlines()
            random.shuffle(lines)
            lines_train = lines[: split_point]
            lines_test = lines[split_point:]
            with open('data/' + file + '.train', 'w') as f2:
                for line in lines_train:
                    line = line.strip()
                    f2.write(line + '\r')
            with open('data/' + file + '.test', 'w') as f2:
                for line in lines_test:
                    line = line.strip()
                    f2.write(line + '\r')


def p2():
    from os import listdir
    from os.path import isfile, join
    path_neg = ['data/aclImdb/test/neg', 'data/aclImdb/train/neg']
    path_pos = ['data/aclImdb/test/pos', 'data/aclImdb/train/pos']
    with open('data/emotion_large_neg', 'w') as f1:
        for path in path_neg:
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            for file in onlyfiles:
                with open(join(path, file), 'r') as f2:
                    line = clean_str(f2.read().strip())
                    f1.write(line+'\n')

    with open('data/emotion_large_pos', 'w') as f1:
        for path in path_pos:
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            for file in onlyfiles:
                with open(join(path, file), 'r') as f2:
                    line = clean_str(f2.read().strip())
                    f1.write(line+'\n')

    file_list = ['emotion_large_neg', 'emotion_large_pos']

    with open('vocb.txt', 'w') as wf:
        voc_dict = {}
        for file in file_list:
            with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
                for word in rf.read().split():
                    if word not in voc_dict:
                        voc_dict[word] = 1
                    else:
                        voc_dict[word] += 1
        idx = 0
        for voc in voc_dict:
            wf.write(voc + ' ' + str(idx) + '\n')
            idx += 1
        wf.write('<unk> ' + str(idx))

p1_1()
