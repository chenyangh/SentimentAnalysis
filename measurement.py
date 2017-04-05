import sklearn.metrics as sm


def meass(y_true, y_pred):
    print('Confusion matrix is\n', sm.confusion_matrix(y_true, y_pred))
    print('F macro is\n', sm.f1_score(y_true, y_pred, average='macro'))
    print('F micro is\n', sm.f1_score(y_true, y_pred, average='micro'))
    print('F1  matrix is\n', sm.classification_report(y_true, y_pred))
