"""
Code for running our Random Forest tests
"""
from sklearn import ensemble, utils
import csv
from numpy import array
from sklearn.cross_validation import StratifiedKFold
from dataset_reader import DatasetReader
from sklearn.metrics import confusion_matrix


def main():
  dataset_reader = DatasetReader("/scratch/cpillsb1/cs66/data/")

  # uncomment for cancer
  # X, y, X_final, y_final, dataset = dataset_reader.load_cancer()

  X, y, X_final, y_final, dataset = dataset_reader.load_higgs()

  skf = StratifiedKFold(y, n_folds=4, shuffle = True, random_state=42)
  nums = [5, 10, 30, 50]
  ii = 0
  best_acc = 0

  for train, test in skf:
      x_train = X[train]
      x_test = X[test]

      y_train = y[train]
      y_test = y[test]

      rf = ensemble.RandomForestClassifier(max_depth=1, n_estimators=nums[ii], random_state=42)

      rf.fit(x_train, y_train)

      predictions = rf.predict(x_test)
      acc = 0.0
      for i, prediction in enumerate(predictions):
        if prediction == y_test[i]:
          acc += 1
      acc /= len(predictions)

      if acc > best_acc:
          best_classifier = rf
          n_estimators = nums[ii]
          best_acc = acc
      ii += 1

  evaluate_test(best_classifier, X_final, y_final, n_estimators, dataset)

"""
Function for evaluating a model on test data. Writes the accuracy and confusion
matrix to a file.
"""
def evaluate_test(clf, X_test, y_test, n_estimators, dataset):
    final_predictions = clf.predict(X_test)
    test_acc = 0.0
    for i, prediction in enumerate(final_predictions):
      if prediction == y_test[i]:
        test_acc += 1
    test_acc /= len(final_predictions)

    with open("../results/rf{}Final.txt".format(dataset), 'w') as f:
      f.write("n_estimators = {}\n".format(str(n_estimators)))
      f.write("Accuracy = {}".format(str(test_acc)))
      f.write("\n\n")
      f.write(str(confusion_matrix(y_test, final_predictions)))
    print "\n\nTest Accuracy: {}".format(test_acc)

if __name__ == '__main__':
  main()
