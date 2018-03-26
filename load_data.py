import json
from datetime import datetime
import re
import pickle
import nltk.data
from nltk.corpus import stopwords

# Part of the code is directly copied from Kaggle's tutorial
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
    print('Reading data finsh', datetime.now())
    return data


def review_to_words(a_review, to_list=True, remove_stopwords=False):
    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                          " ",                   # The pattern to replace it with
                          a_review)  # The text to search

    # Removing stop words
    lower_case = letters_only.lower()
    words = lower_case.split()
    if remove_stopwords:
        words = [w for w in words if w not in stopwords.words("english")]
    # return words
    if to_list:
        return words
    else:
        return " ".join(words)


def review_to_sentences(review, __remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_words(raw_sentence, remove_stopwords=__remove_stopwords))
    return sentences


def generate_date_without_punkt():
    file_name = 'data/Digital_Music_5.json'
    data = read_data(file_name)
    clean_data = []
    count = 0
    for item in data:
        count += 1
        a_review_words = review_to_words(item[0])
        clean_data.append([a_review_words, item[1]])
        if count % 1000 == 0:
            print("Processed review", count)

    f = open('clean_data_string', 'wb')
    pickle.dump(clean_data, f)
    f.close()


def generate_data_with_punkt():
    file_name = 'data/Digital_Music_5.json'
    data = read_data(file_name)
    clean_data = []
    count = 0
    "Parsing sentences from training set"
    for item in data:
        a_review_words = review_to_sentences(item[0],
                                             __remove_stopwords=True)
        clean_data.append([a_review_words, item[1]])
        if count % 1000 == 0:
            print("Processed review", count)
        count += 1

    f = open('data/clean_data_using_stopwords_punkt', 'wb')
    pickle.dump(clean_data, f)
    f.close()

generate_data_with_punkt()
