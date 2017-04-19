import os
import re
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from collections import Counter
import logging
from gensim.models import word2vec
import gensim

# logging.getLogger().setLevel(logging.INFO)
import re
import nltk

file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
embedding = 'cbow'  # optional senti | cbow | skip


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


def sentence_to_word_list(a_review):
    # Use regular expressions to do a find-and-replace

    tmp = a_review.split()
    words = []
    for word in tmp:
        if False:
            words.extend(clean_str(word).split())
        else:
            words.extend(remove_punctuation(word).split())
    return words


def train_word2vec():
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    sentences = get_sentences()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 9  # Number of threads to run in parallel
    context = 5  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model skip_gram...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, sg=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "feature/imdb_skip_gram_word2vec"
    model.save(model_name)
    print("Training model cbow...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling, sg=0)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "feature/imdb_cbow_word2vec"
    model.save(model_name)


def get_sentences():
    sentences = []
    for file in file_list:
        with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
            tmp = rf.readlines()
            for line in tmp:
                sentences.extend(split_sentences(line))
    return sentences


def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9]", " ", s)
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


def remove_punctuation(s):
    s = re.sub(r"[^A-Za-z0-9]", " ", s)
    return s.strip().lower()


def load_fast_text(vocabulary):
    print('start loading fasttext')
    #model = gensim.models.KeyedVectors.load_word2vec_format('feature/facebookfasttext.vec')
    with open('feature/fasttextModel', 'br') as f:
        model = pickle.load(f)
    print('finish loading fasttext')
    embed_dict = model.vocab
    word_embeddings = {}
    num_oov = 0
    for word in vocabulary:
        if word in embed_dict:
            vec = model.syn0[embed_dict[word].index]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1

        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, 300)
    print('Numb of oov is', num_oov)
    return word_embeddings


def load_glove(vocabulary):
    print('start loading glove')
    # model = gensim.models.KeyedVectors.load_word2vec_format('feature/glove.840B.300d.vec')
    with open('feature/gloveModel', 'br') as f:
        model = pickle.load(f)
    print('finish loading glove')
    embed_dict = model.vocab
    word_embeddings = {}
    num_oov = 0
    for word in vocabulary:
        if word in embed_dict:
            vec = model.syn0[embed_dict[word].index]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1

        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, 300)
    print('Numb of oov is', num_oov)
    return word_embeddings


def load_word2vec(vocabulary):
    print('start loading word2vec')
    # model = gensim.models.KeyedVectors.load_word2vec_format('feature/GoogleNews-vectors-negative300.bin', binary=True)
    with open('feature/word2vecModel', 'br') as f:
        model = pickle.load(f)
    print('finish loading word2vec')
    embed_dict = model.vocab
    word_embeddings = {}
    num_oov = 0
    for word in vocabulary:
        if word in embed_dict:
            vec = model.syn0[embed_dict[word].index]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1

        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, 300)
    print('Numb of oov is', num_oov)
    return word_embeddings


def load_random_word2vec(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-1, 1, 300)
    return word_embeddings


def load_my_word2vec(vocabulary):
    model_name = os.path.join('feature', 'imdb_cbow_word2vec')
    model = word2vec.Word2Vec.load(model_name)
    embed_dict = model.wv.vocab
    word_embeddings = {}
    num_oov = 0
    for word in vocabulary:
        if word in embed_dict:
            vec = model.wv.syn0[embed_dict[word].index]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1

        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, 300)
    print('Numb of oov is', num_oov)
    return word_embeddings


def load_senti_emb(vocabulary):
    #pretrained_word2vec_embbeding = load_word2vec(vocabulary)

    sent_emb_dict = {}
    with open('feature/SSWE/-round-220', 'r') as f:
        for line in f.readlines():
            line = line.split()
            word = line[0]
            vector = [np.float32(x) for x in line[1:]]
            sent_emb_dict[word] = vector

    word_embeddings = {}
    num_oov = 0
    word_in_sswe = 0
    for word in vocabulary:
        if word in sent_emb_dict:
            word_in_sswe += 1
            vec = sent_emb_dict[word]
            word_embeddings[word] = (vec - min(vec)) / np.add(max(vec), -min(vec)) * 2 - 1
        # elif word in pretrained_word2vec_embbeding:
        #     word_embeddings[word] = pretrained_word2vec_embbeding[word]
        else:
            num_oov += 1
            word_embeddings[word] = np.random.uniform(-1, 1, 300)
    print('Word in SSWE is', word_in_sswe, 'Total num of oov is', num_oov)
    return word_embeddings


def load_embeddings(vocabulary):

    word_embeddings = load_glove(vocabulary)

    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None, given_pad_len=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if given_pad_len is not None:
            sequence_length = given_pad_len
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
def load_imdb():
    x_raw = []
    y_raw = []

    file = file_list[0]  # neg
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([1, 0])
            x_raw.append(sentence_to_word_list(line))

    file = file_list[1]  # pos
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([0, 1])
            x_raw.append(sentence_to_word_list(line))
    return x_raw, y_raw


def load_imdb_train():
    file_list_train = ['emotion.neg.0.txt.train', 'emotion.pos.0.txt.train']
    x_raw = []
    y_raw = []

    file = file_list_train[0]  # neg
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([1, 0])
            x_raw.append(sentence_to_word_list(line))

    file = file_list_train[1]  # pos
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([0, 1])
            x_raw.append(sentence_to_word_list(line))
    return x_raw, y_raw


def load_imdb_test():
    file_list_test = ['emotion.neg.0.txt.test', 'emotion.pos.0.txt.test']
    x_raw = []
    y_raw = []

    file = file_list_test[0]  # neg
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([1, 0])
            x_raw.append(sentence_to_word_list(line))

    file = file_list_test[1]  # pos
    with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
        for line in rf.readlines():
            y_raw.append([0, 1])
            x_raw.append(sentence_to_word_list(line))
    return x_raw, y_raw


def load_data():
    x_raw, y_raw = load_imdb()
    x_raw, sequence_length = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)
    x_train_dev, y_train_dev = load_imdb_train()
    x_train_dev, _ = pad_sentences(x_train_dev, given_pad_len=sequence_length)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_train_dev])
    y = np.array(y_train_dev)

    x_test, y_test = load_imdb_test()
    x_test, _ = pad_sentences(x_test, given_pad_len=sequence_length)
    x_test = np.array([[vocabulary[word] for word in sentence] for sentence in x_test])
    y_test = np.array(y_test)

    labels = ['negative', 'positive']
    return x, y, x_test, y_test, vocabulary, vocabulary_inv, labels


if __name__ == "__main__":
    train_word2vec()
