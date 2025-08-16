# functions for training model and tuning parameters

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV


# training model
def train_model(X_train_encoded, y_train, X_test_encoded, y_test):
    clf = RandomForestClassifier(n_estimators=80, random_state=3)
    clf.fit(X_train_encoded, y_train)
    return clf


# tuning n estimators
def tune_estimators(X_train_encoded, y_train, X_test_encoded, y_test):
    np.random.seed(3)

    for i in range(10, 100, 10):
        print(f"Trying {i} estimators:")
        clf = RandomForestClassifier(n_estimators=i, random_state=3).fit(X_train_encoded, y_train)
        print(f"score: {clf.score(X_test_encoded, y_test)}")
        print("")


# tuning other hyperparameters
def tune(X_train_encoded, y_train):

    param_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_leaf': [1, 25, 50, 75, 100, 200],
        'min_samples_split': [2, 5],
        'min_impurity_decrease': [0.0, 0.001, 0.005]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=3, n_estimators=80), param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid.fit(X_train_encoded, y_train)

    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)