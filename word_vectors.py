import pickle
import numpy as np
import logging
from gensim.models import word2vec
from sklearn.cluster import KMeans
import time
from sklearn.ensemble import RandomForestClassifier

file_name = 'data/clean_data_no_stopwords_punkt'
f = open(file_name, 'rb')
clean_data = pickle.load(f)
f.close()

clean_data = np.array(clean_data)

# Split test and training
num_data = len(clean_data)
num_test = 5000
train_data = clean_data[: num_data - num_test]
sentences = [item for sublist in train_data[:, 0] for item in sublist]


def train_word2vec():
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 9       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                size=num_features, min_count=min_word_count,
                window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "feature/300features_40minwords_10context"
    model.save(model_name)


def having_fun_with_word2vec():
    model_name = "feature/300features_40minwords_10context"
    model = word2vec.Word2Vec.load(model_name)
    # print(model.most_similar("terrible")) # beautiful

def K_means_train():
    start = time.time()  # Start time
    model_name = "feature/300features_40minwords_10context"
    model = word2vec.Word2Vec.load(model_name)
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 10)

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")

    __f = open('feature/vector-divby10-means', 'wb')
    pickle.dump(idx, __f)
    __f.close()


def create_bag_of_centroids(wordlist, word_centroid_map, num_clusters):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map

    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_clusters, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def random_forest_train():
    __f = open('feature/vector-divby10-means', 'rb')
    idx = pickle.load(__f)
    __f.close()
    model_name = "feature/300features_40minwords_10context"
    model = word2vec.Word2Vec.load(model_name)
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    __file_name = 'data/clean_data_wordlist'
    __f = open(__file_name, 'rb')
    __clean_data = pickle.load(__f)
    __f.close()

    __clean_data = np.array(__clean_data)

    # Split test and training
    __num_data = len(__clean_data)
    __num_test = 5000
    __train_data = __clean_data[: __num_data - __num_test]
    __test_data = __clean_data[__num_data - __num_test:]

    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 10)

    train_centroids = np.zeros((len(__train_data), num_clusters),
                               dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in __train_data:
        train_centroids[counter] = create_bag_of_centroids(review[0],
                                            word_centroid_map, num_clusters)
        counter += 1

    # Transform the test set reviews into bags of centroids
    test_centroids = np.zeros((len(__test_data), num_clusters),
                              dtype="float32")
    counter = 0
    for review in __test_data:
        test_centroids[counter] = create_bag_of_centroids(review[0],
                                            word_centroid_map, num_clusters)
        counter += 1

    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    train_label = np.asarray(__train_data[:, 1], dtype="|S6")
    forest = forest.fit(train_centroids, train_label)
    __file_name = 'model/wordvector_randomforest'
    __f = open(__file_name, 'wb')
    pickle.dump(forest, __f)
    __f.close()

    test_label = np.asarray(__test_data[:, 1], dtype="|S6")
    result = forest.predict(test_centroids)
    count = 0
    for i in range(len(test_label)):
        if train_label[i] == result[i]:
            count += 1
    print(train_label, count / len(test_label))

random_forest_train()
