import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

f = open('data/clean_data2', 'rb')
clean_data = pickle.load(f)
f.close()
print('The total size of data set is', len(clean_data))


vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                             preprocessor=None, stop_words=None,
                             max_features=2000)


# Splitting training and testing data.
data_array = np.array(clean_data)
data_size = data_array.shape[0]
train_size = data_size - 5000  # select a test size of 5000
data_train = data_array[:train_size]
data_test = data_array[train_size:]
review_to_train = data_train[:, 0]
label_to_train = data_train[:, 1]
review_to_test = data_test[:, 0]
label_to_test = data_test[:, 1]


IF_TRAIN = True  #
if IF_TRAIN:
    # Prepare train data
    train_data_features = vectorizer.fit_transform(review_to_train)
    train_data_features = train_data_features.toarray()

    print("Starting training Random Forest")
    n_estimators = 100
    forest = RandomForestClassifier(n_estimators)
    forest = forest.fit(train_data_features, label_to_train)
    print("Finish training Random Forest")
    f = open('model/trained_forest_2000_' + str(n_estimators), 'wb')
    pickle.dump(forest, f)
    f.close()
    print('The total size of data set is', len(clean_data))
else:
    f = open('model/trained_forest500', 'rb')
    forest = pickle.load(f)
    f.close()


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
