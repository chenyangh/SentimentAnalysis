from load_data import read_data
import re
from nltk.corpus import stopwords
import pickle
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
    return words


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


f = open('clean_data', 'wb')
pickle.dump(clean_data, f)
f.close()


