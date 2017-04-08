from gensim.models import word2vec
import logging
import re
import nltk

file_list = ['emotion.neg.0.txt', 'emotion.pos.0.txt']
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_voc_file():
    with open('vocb.txt', 'w') as wf:
        voc_dict = {}
        for file in file_list:
            with open('data/'+file, 'r', encoding="ISO-8859-1") as rf:
                for word in rf.read().split():
                    if word not in voc_dict:
                        voc_dict[word] = 1
                    else:
                        voc_dict[word] += 1
        for voc in voc_dict:
            wf.write(voc + ' ' + str(voc_dict[voc]) + '\n')


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
    letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                          " ",                   # The pattern to replace it with
                          a_review)  # The text to search

    # Removing stop words
    lower_case = letters_only.lower()
    words = lower_case.split()

    return words


def get_sentences():
    sentences = []
    for file in file_list:
        with open('data/' + file, 'r', encoding="ISO-8859-1") as rf:
            for line in rf.readlines():
                sentences.extend(split_sentences(line))
    return sentences


def train_word2vec():
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    sentences = get_sentences()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 500    # Word vector dimensionality
    min_word_count = 20   # Minimum word count
    num_workers = 9       # Number of threads to run in parallel
    context = 3          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model skip_gram...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                size=num_features, min_count=min_word_count,
                window=context, sample=downsampling,sg=1)

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

if __name__ == '__main__':
    train_word2vec()