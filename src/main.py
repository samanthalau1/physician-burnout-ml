# full functionality of ml physician burnout project from start to end

from data_utils import load_data, clean, extractNum, split, encode
from plot import level_count, hours_correlation, years_correlation
from train import train_model, tune_estimators, tune
from evaluate import print_stats, conf_matrix, predict_error

# loading and cleaning data
file_path = r'C:\Users\saman\physician-burnout-ml\Physician Burnout Survey.xlsx'
data = load_data(file_path)
clean_data = clean(data)

# burnout level count visualization
level_count(clean_data)

# removing words from burnout level
clean_data['Burnout Level'] = clean_data['Burnout Level'].apply(extractNum)

# correlation visualizations
hours_correlation(clean_data)
years_correlation(clean_data)

X_train, X_test, y_train, y_test = split(clean_data)
X_train_encoded, X_test_encoded = encode(X_train, X_test)

# training model
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
clf = train_model(X_train_encoded, y_train, X_test_encoded, y_test)
clf.score(X_test_encoded, y_test)

# model evaluation
print_stats(X_test_encoded, y_test, clf)

# hyperparameter tuning
tune_estimators(X_train_encoded, y_train, X_test_encoded, y_test)
tune(X_train_encoded, y_train)

# evaluation visualizations
conf_matrix(X_train_encoded, y_train, X_test_encoded, y_test, clf)
predict_error(X_train_encoded, y_train, X_test_encoded, y_test, clf)