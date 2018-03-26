import pickle
import numpy as np
f = open('data/clean_data_no_stopwords_punkt', 'rb')
clean_data = pickle.load(f)
f.close()

clean_data = np.array(clean_data)
# Split test and training
num_data = len(clean_data)
num_test = 5000
train_data = clean_data[:num_data - num_test]
test_data = clean_data[num_data - num_test:]


def words_and_sentences_count():
    print("There are ", len(clean_data), "sentences")

    sentences_train = [item for sublist in train_data[:, 0] for item in sublist]
    sentences_test = [item for sublist in test_data[:, 0] for item in sublist]

    unique_words_train = {}
    num_words = 0
    for sentence in sentences_train:
        for word in sentence:
            num_words += 1
            if word not in unique_words_train:
                unique_words_train[word] = 1
            else:
                unique_words_train[word] += 1

    print("In training: there are", len(sentences_train), "sentences,",
          len(unique_words_train), "unique words", num_words, "words over all")

    unique_words_test = {}
    num_words = 0
    for sentence in sentences_test:
        for word in sentence:
            num_words += 1
            if word not in unique_words_test:
                unique_words_test[word] = 1
            else:
                unique_words_test[word] += 1

    print("In testing: there are", len(sentences_test), "sentences,",
          len(unique_words_test), "unique words", num_words, "words over all")

    unknown = 0
    for word in unique_words_test:
        if word not in unique_words_train:
            unknown += 1

    print(unknown, "testing words are not in training")


def balance_of_data():
    counts_train = {}
    __num_train = 0
    for review in train_data:
        __num_train += 1
        if review[1] not in counts_train:
            counts_train[review[1]] = 1
        else:
            counts_train[review[1]] += 1
    print("Training set:")
    for item in counts_train:
        counts_train[item] = counts_train[item] / __num_train
        print(item, counts_train[item])

    counts_test = {}
    __num_test = 0
    for review in test_data:
        __num_test += 1
        if review[1] not in counts_test:
            counts_test[review[1]] = 1
        else:
            counts_test[review[1]] += 1

    print("Testing set:")
    for item in counts_test:
        counts_test[item] = counts_test[item] / __num_test
        print(item, counts_test[item])

balance_of_data()
