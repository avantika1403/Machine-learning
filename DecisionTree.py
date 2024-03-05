# Decision tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
def read_datasets():
    print("---------------------------------------------------")
    datasets = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
                           sep=",", header=None)
    print(f"Dataset Length : {len(datasets)}")
    print(f"Dataset Shape: {datasets.shape}")
    print(f"Datasets : {datasets.head()}")
    return datasets

def splitdataset(datasets):
    X = datasets.values[:, 1:5]
    Y = datasets.values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test
def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print(f"Predicted values: {y_pred}")
    return y_pred
def cal_accuracy(y_test, y_pred):
    print(f"Confusion Metrix :{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}")
    print(f"Report: {classification_report(y_test, y_pred)}")
def main():
    datasets = read_datasets()
    X_train, X_test, y_train, y_test = splitdataset(datasets)
    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)
    print("Results using Gini Index: ")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    print("Results using Entropy Index: ")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
if __name__ == "__main__":
    main()