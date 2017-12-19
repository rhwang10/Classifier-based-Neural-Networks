"""
Class for reading the two different datasets
"""

import csv
from sklearn.datasets import load_breast_cancer
from numpy import array
from sklearn import utils


class DatasetReader(object):
    def __init__(self, path):
        self.path = path

    """
    Loads the cancer dataset
    @returns, x_train, y_train, x_test, y_test, dataset_name
    """
    def load_cancer(self):
        dataset = load_breast_cancer()
        X, y = dataset["data"], dataset["target"]
        return X[100:], y[100:], X[:100], y[:100], "cancer"

    """
    Loads the higgs dataset
    @returns, x_train, y_train, x_test, y_test, dataset_name
    """
    def load_higgs(self):
        X = []
        #get train data
        with open("{}{}".format(self.path,"training.csv"), 'rb') as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          for row in reader:
            X.append(row)

        #split train into data and labels, also shuffle and subsample
        all_labels = X.pop(0)
        all_labels.pop(0)
        y = []
        for example in X:
          example.pop(0) #take out the eventID feature
          y.append(example.pop()) #take out the label
          example.pop()
        X,y = utils.shuffle(X,y, random_state=42)

        X_test = array(X[:50000], dtype="float")
        print X_test.shape
        y_test = y[:50000]
        y=y[50000:]
        X = array(X[50000:], dtype='float')
        y = array([1.0 if y_val == 's' else 0.0 for y_val in y])
        y_test = array([1.0 if y_val == 's' else 0.0 for y_val in y_test])

        return X, y, X_test, y_test, "higgs"
