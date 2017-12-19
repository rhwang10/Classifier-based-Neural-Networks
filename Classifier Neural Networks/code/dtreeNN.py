"""
Code for testing our dtreeNN implementation
"""
from network import Network
from sklearn import utils
from sklearn.datasets import load_breast_cancer
import numpy as np
from layer import Layer
from sklearn.tree import ExtraTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import csv
from numpy import array
from sklearn.metrics import confusion_matrix
from dataset_reader import DatasetReader


def main():
    # Enter path to higgs dataset
    dataset_reader = DatasetReader("/scratch/cpillsb1/cs66/data/")


    X, y, X_final, y_final, dataset = dataset_reader.load_cancer()

    # uncomment for higgs
    # X, y, X_final, y_final, dataset = dataset_reader.load_higgs()


    skf = StratifiedKFold(y, n_folds=4, shuffle = True, random_state=42)

    ii = 0
    for train, test in skf:
        x_train = X[train]
        x_test = X[test]

        y_train = y[train]
        y_test = y[test]
        nums = [5, 10, 30, 50]
        layer = Layer(ExtraTreeClassifier, {"max_depth": 1}, x_train, y_train, nums[ii])
        predictions = layer.predictAll(x_train)
        lr = Layer(LogisticRegression, {"n_jobs":-1, "max_iter":1000}, predictions, y_train, 1)
        network = Network( [layer, lr] )

        evaluate_test(network, X_final, y_final, nums[ii], dataset)

        ii += 1

"""
Function for evaluating a model on test data. Writes the accuracy and confusion
matrix to a file
"""
def evaluate_test(clf, X_test, y_test, num_nodes, dataset):
    final_predictions = clf.predictAll(X_test)
    test_acc = 0.0
    for i, prediction in enumerate(final_predictions):
      if prediction == y_test[i]:
        test_acc += 1
    test_acc /= len(final_predictions)
    with open("../results/dtreeNN{}{}Final.txt".format(str(num_nodes), dataset), 'w') as f:
      f.write("num_nodes = {}\n".format(str(num_nodes)))
      f.write("Accuracy = {}".format(str(test_acc)))
      f.write("\n\n")
      f.write(str(confusion_matrix(y_test, final_predictions)))
    print "\n\nTest Accuracy: {}".format(test_acc)

if __name__ == "__main__":
    main()
