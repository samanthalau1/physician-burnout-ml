# functions for evaluating performance of the model

from sklearn.metrics import classification_report
from yellowbrick.classifier import ConfusionMatrix, ClassPredictionError


# prints classification report of model performance
def print_stats(X_test_encoded, y_test, clf):
    y_preds = clf.predict(X_test_encoded)
    print(classification_report(y_test, y_preds))


# graph of confusion matrix
def conf_matrix(X_train_encoded, y_train, X_test_encoded, y_test, clf):
    classes = ['Low', 'Moderate', 'High']
    cm = ConfusionMatrix(
    clf, classes=classes,
    percent=True, label_encoder={0: 'Low', 1: 'Moderate', 2: 'High'})

    cm.fit(X_train_encoded, y_train)
    cm.score(X_test_encoded, y_test)

    cm.show()

    for label in cm.ax.texts:
        label.set_size(22)


# graph of class prediction error
def predict_error(X_train_encoded, y_train, X_test_encoded, y_test, clf):
    classes = ['Low', 'Moderate', 'High']
    visualizer = ClassPredictionError(clf, classes=classes)
    visualizer.fit(X_train_encoded, y_train)
    visualizer.score(X_test_encoded, y_test)
    visualizer.show()