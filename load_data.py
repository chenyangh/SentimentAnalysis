import json
from datetime import datetime
import re
from nltk.corpus import stopwords
import pickle

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



# data is the global variable

file_name = 'Digital_Music_5.json'
data = read_data(file_name)


def review_to_words(a_review):
    # Use regular expressions to do a find-and-replace
    letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                          " ",                   # The pattern to replace it with
                          a_review)  # The text to search

    # Removing stop words
    lower_case = letters_only.lower()
    words = lower_case.split()
    words = [w for w in words if w not in stopwords.words("english")]
    # return words
    return " ".join(words)


# Data pre-processing
a_review = data[0][0]


clean_data = []

count = 0
for item in data:
    count += 1
    a_review_words = review_to_words(item[0])
    clean_data.append([a_review_words, item[1]])
    if count % 1000 == 0:
        print("Processed review", count)


f = open('clean_data2', 'wb')
pickle.dump(clean_data, f)
f.close()

