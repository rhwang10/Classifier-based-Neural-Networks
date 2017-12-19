from keras.models import *
from keras.layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import StratifiedKFold
from keras.optimizers import *
from keras.initializers import *
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from dataset_reader import DatasetReader
from numpy import array
from sklearn.metrics import confusion_matrix


def main():

    dataset_reader = DatasetReader("/scratch/cpillsb1/cs66/data/")

    # uncomment for cancer
    # X, y, X_final, y_final, dataset = dataset_reader.load_cancer()

    X, y, X_final, y_final, dataset = dataset_reader.load_higgs()

    input_s = (30,)
    batch_size = 25
    classes = 2

    num_nodes = [5,10,30,50]

    skf = StratifiedKFold(y, n_folds=4, shuffle = True, random_state=42)
    best_acc = 0
    ii = 0

    for train,test in skf:
        x_train = X[train]
        x_test = X[test]

        y_train = y[train]
        y_test = y[test]

        y_train = to_categorical(y_train, classes)
        y_test = to_categorical(y_test, classes)

        neural_net = Sequential()

        neural_net.add(Dense(num_nodes[ii], activation='sigmoid', input_shape = input_s, kernel_initializer="TruncatedNormal"))
        neural_net.add(Dropout(.01))
        neural_net.add(Dense(2, activation='softmax'))

        neural_net.compile(optimizer="RMSProp", loss = 'binary_crossentropy', metrics = ['accuracy'])

        neural_net.fit(x_train, y_train, batch_size = batch_size, epochs = 100, verbose = 0, validation_data = (x_test, y_test))

        predictions = neural_net.predict(x_test)
        predictions = [round(x[1]) for x in predictions]
        y_test = [x[1] for x in y_test]

        acc = 0.0
        for i, prediction in enumerate(predictions):
          if prediction == y_test[i]:
            acc += 1
        acc /= len(predictions)

        if acc > best_acc:
            best_classifier = neural_net
            best_num_nodes = num_nodes[ii]
            best_acc = acc

        ii += 1

    evaluate_test(best_classifier, X_final, y_final, best_num_nodes, dataset)

"""
Function for evaluating a model on test data. Writes the accuracy and confusion
matrix to a file
"""
def evaluate_test(clf, X_test, y_test, num_nodes, dataset):
  final_predictions = clf.predict(X_test)
  final_predictions = [round(x[1]) for x in final_predictions]
  test_acc = 0.0

  for i, prediction in enumerate(final_predictions):
    if prediction == y_test[i]:
      test_acc += 1
  test_acc /= len(final_predictions)

  with open("../results/nn{}Final.txt".format(dataset), 'w') as f:
    f.write("num_nodes = {}\n".format(str(num_nodes)))
    f.write("Accuracy = {}".format(str(test_acc)))
    f.write("\n\n")
    f.write(str(confusion_matrix(y_test, final_predictions)))
  print "\n\nTest Accuracy: {}".format(test_acc)



if __name__ == "__main__":
    main()
