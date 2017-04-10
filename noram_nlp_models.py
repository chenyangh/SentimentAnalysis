from data_helper import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from data_helper 

def create_bow_feature():
    x_raw, y_raw = load_imdb()
    y_raw_2 = []
    for item in y_raw:
        if item == [0, 1]:
            y_raw_2.append(1)
        else:
            y_raw_2.append(0)
    x_raw_2 = [' '.join(x) for x in x_raw]

    x, x_test, y, y_test = train_test_split(x_raw_2,  y_raw_2, test_size=0.1)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

    bag_of_words_len = 400
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words=stopwords.words("english"),
                                 max_features=bag_of_words_len)
    # Prepare train data
    x_train_fea = vectorizer.fit_transform(x_train)
    x_train_fea = x_train_fea.toarray()
    x_test_fea = vectorizer.transform(x_test)
    x_test_fea = x_test_fea.toarray()


def create_emb_feature():
    pass


def random_forest(x_train_fea, y_train, x_test_fea, y_test):
    print("Starting training Random Forest")
    n_estimators = 100
    forest = RandomForestClassifier(n_estimators)
    forest = forest.fit(x_train_fea, y_train)
    print("Finish training Random Forest")
    # Testing
    test_result = forest.predict(x_test_fea)

    return accuracy_score(y_test, test_result)


# NB
def nb(x_train_fea, y_train, x_test_fea, y_test):
    nb_clf = MultinomialNB()
    nb_clf.fit(x_train_fea, y_train)
    test_result = nb_clf.predict(x_test_fea)
    return accuracy_score(y_test, test_result)


# NB + tfidf
def nb_tfidf(x_train_fea, y_train, x_test_fea, y_test):
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_fea)
    train_tf = tf_transformer.transform(x_train_fea)
    clf_tf = MultinomialNB().fit(train_tf, y_train)
    test_tf = tf_transformer.transform(x_test_fea)
    test_result_tf = clf_tf.predict(test_tf)
    return accuracy_score(y_test, test_result_tf)


def train_svm(x_train_fea, y_train, x_test_fea, y_test):
    svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    svm_clf.fit(x_train_fea, y_train)
    result = svm_clf.predict(x_test_fea)
    return accuracy_score(y_test, result)
