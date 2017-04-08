# Create own data set
from datetime import datetime
import json
import pickle
import nltk
import re

# Read the json file
# def read_data(file_name=''):
#     data = []
#     print('Reading data start', datetime.now())
#     # file_name = 'Electronics_5.json'  # 13 secs loading time
#     # file_name = 'Digital_Music_5.json'  # 1 sec loading time
#
#     f = open(file_name, 'r')
#     for line in f.readlines():
#         tmp = json.loads(line)
#         data.append([tmp['reviewText'], tmp['overall']])
#     f.close()
#     print('Reading data finsh', datetime.now())
#     return data
# file_name = 'data/books.pkl'

# data = read_data(file_name)

# with open('data/books.pkl', 'bw') as f:
#     pickle.dump(data, f)
# del data



# Create splitted sentences for training word2vec
file_name = 'data/books.pkl'
with open(file_name, 'br') as f:
    data = pickle.load(f)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


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


def sentence_to_word_list(a_review):
    # Use regular expressions to do a find-and-replace

    tmp = a_review.split()
    words = []
    for word in tmp:
        words.extend(clean_str(word).split())

    return words


def split_sentences(review, __remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        tmp = sentence_to_word_list(raw_sentence)
        if len(tmp) > 0:
            sentences.append(tmp)
    return sentences


import multiprocessing


def work(idx):
    __i = idx
    print('Starting thread', str(__i + 1))
    sentences = []
    count = 0
    batch = 2000000
    for item in range(batch * __i, min(batch * (__i + 1), len(data))):
        if count % 10000 == 1:
            print('Thread', str(__i + 1), 'at', count, 'out of',
                  str(min(batch * (__i + 1), len(data))-batch*__i))
        count += 1
        sent = data[item][0]
        sentences.extend(split_sentences(sent))

    with open('data/book_sents_' + str(__i) + '.pkl', 'bw') as f:
        pickle.dump(sentences, f)
    del sentences
    print('Finish thread', str(__i + 1))


try:
    pool = multiprocessing.Pool(5)
    total_tasks = 5
    tasks = range(total_tasks)
    results = pool.map_async(work, tasks)
    pool.close()
    pool.join()
except:
    print("Error: unable to start thread")


from gensim.models.word2vec import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 300    # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 12       # Number of threads to run in parallel
context = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
i = 0
print('loading', str(i), 'th file')
with open('data/book_sents_' + str(i) + '.pkl', 'br') as f:
    sentences = pickle.load(f)

print("Training model skip_gram...")
model = Word2Vec(sentences, workers=num_workers,
            size=num_features, min_count=min_word_count,
            window=context, sample=downsampling, sg=1)
del sentences
print('finish', str(i), 'th file')

for i in range(1, 5):
    print('loading', str(i), 'th file')
    with open('data/book_sents_' + str(i) + '.pkl', 'br') as f:
        sentences = pickle.load(f)
    print("Training model skip_gram...")
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count)
    print('finish', str(i), 'th file')
    del sentences

with open('data/book_skip_gram2', 'bw') as f:
    pickle.dump(model, f)
