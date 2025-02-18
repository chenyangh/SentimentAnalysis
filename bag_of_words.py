import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


f = open('data/clean_data_string', 'rb')
clean_data = pickle.load(f)
f.close()
print('The total size of data set is', len(clean_data))


# Splitting training and testing data.
data_array = np.array(clean_data)
data_size = data_array.shape[0]
train_size = data_size - 5000  # select a test size of 5000
data_train = data_array[:train_size]
data_test = data_array[train_size:]
review_to_train = data_train[:, 0]
label_to_train = np.asarray(data_train[:, 1], dtype="|S6")
review_to_test = data_test[:, 0]
label_to_test = np.asarray(data_test[:, 1], dtype="|S6")


def train_random_forest():
    IF_TRAIN = True  #
    if IF_TRAIN:
        bag_of_words_len = 5000
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                     preprocessor=None, stop_words=None,
                                     max_features=bag_of_words_len)
        # Prepare train data
        train_data_features = vectorizer.fit_transform(review_to_train)
        train_data_features = train_data_features.toarray()

        print("Starting training Random Forest")
        n_estimators = 100
        forest = RandomForestClassifier(n_estimators)
        forest = forest.fit(train_data_features, label_to_train)
        print("Finish training Random Forest")
        __f = open('model/trained_forest_'+str(bag_of_words_len)+'_'+str(n_estimators), 'wb')
        pickle.dump(forest, __f)
        __f.close()
        print('The total size of data set is', len(clean_data))
    else:
        __f = open('model/trained_forest_2000_500', 'rb')
        forest = pickle.load(__f)
        __f.close()

    # Testing:
    # Prepare test data
    test_data_features = vectorizer.fit_transform(review_to_test)
    test_data_features = test_data_features.toarray()

    print("Starting testing")
    test_result = forest.predict(test_data_features)
    count = 0
    for i in range(label_to_test.size):
        if label_to_test[i] == test_result[i]:
            count += 1
    print(count / len(label_to_test))


def train_naive_bayes():
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(review_to_train)

    # TF model
    tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
    train_tf = tf_transformer.transform(train_counts)
    clf_tf = MultinomialNB().fit(train_tf, label_to_train)
    test_counts_tf = count_vect.transform(review_to_test)
    test_tf = tf_transformer.transform(test_counts_tf)
    test_result_tf = clf_tf.predict(test_tf)

    # Get accuracy
    count = 0
    for i in range(label_to_test.size):
        if label_to_test[i] == test_result_tf[i]:
            count += 1
    print('TF Accuracy', count / len(label_to_test))

    # TF IDF model
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    clf_tfidf = MultinomialNB().fit(train_tfidf, label_to_train)
    test_counts_tfidf = count_vect.transform(review_to_test)
    test_tfidf = tfidf_transformer.transform(test_counts_tfidf)
    test_result_idf = clf_tfidf.predict(test_tfidf)

    # Get accuracy
    count = 0
    for i in range(label_to_test.size):
        if label_to_test[i] == test_result_idf[i]:
            count += 1
    print('TF-IDF Accuracy', count / len(label_to_test))


train_random_forest()
train_naive_bayes()
