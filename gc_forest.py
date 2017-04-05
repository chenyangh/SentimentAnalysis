from gcForest.GCForest import gcForest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4)

gcf = gcForest(shape_1X=[8, 8], window=[4, 6], cascade_layer=5,  min_samples=7)
gcf.fit(X_tr, y_tr)

pred_X = gcf.predict(X_te)
print(pred_X)

accuracy = accuracy_score(y_true=y_te, y_pred=pred_X)
print('gcForest accuracy : {}'.format(accuracy))
