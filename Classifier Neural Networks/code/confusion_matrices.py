"""
Code for generating confusion matrices graphs and getting precision and recall
"""
from sklearn.metrics import confusion_matrix, recall_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools



def main():
    higgs_labels = ("background","signal")
    cancer_labels = ("benign", "malignant")
    precision = {}
    recall = {}

    # Decision Trees
    cm_dtree_higgs = np.array([[27494, 5362], [5980, 11164]])
    precision["Decision Tree Higgs"] = 11164.0 / (11164 + 5362)
    recall["Decision Tree Higgs"] = 11164.0 / (11164 + 5980)
    plot_confusion_matrix(cm_dtree_higgs, higgs_labels, title="Decision Tree Higgs")

    cm_dtree_cancer = np.array([[57, 8], [4, 31]])
    precision["Decision Tree Cancer"] = 31.0 / (31 + 8)
    recall["Decision Tree Cancer"] = 31.0 / (31 + 4)
    plot_confusion_matrix(cm_dtree_cancer, cancer_labels, title="Decision Tree Cancer")

    # Random Forests
    cm_rf_higgs = np.array([[30588, 2477], [10666, 6269]])
    precision["Random Forest Higgs"] = 6269.0 / (6269 + 2477)
    recall["Random Forest Higgs"] = 6269.0 / (6269 + 10666)
    plot_confusion_matrix(cm_rf_higgs, higgs_labels, title="RF Higgs")

    cm_rf_cancer = np.array([[45, 20], [0, 35]])
    precision["Random Forest Cancer"] = 35.0 / (35 + 20)
    recall["Random Forest Cancer"] = 35.0 / (35 + 0)
    plot_confusion_matrix(cm_rf_cancer, cancer_labels, title="RF Cancer")

    # Neural Net
    cm_nn_higgs = np.array([[28814, 4251], [7621, 9314]])
    precision["Neural Net Higgs"] = 9314.0 / (9314 + 4251)
    recall["Neural Net Higgs"] = 9314.0 / (9314 + 7621)
    plot_confusion_matrix(cm_nn_higgs, higgs_labels, title="NN Higgs")

    cm_nn_cancer = np.array([[51, 14], [1, 34]])
    precision["Neural Net Cancer"] = float(34) / (34 + 14)
    recall["Neural Net Cancer"] = float(34) / (34 + 1)
    plot_confusion_matrix(cm_nn_cancer, cancer_labels, title="NN Cancer")

    # RFNN
    cm_rfnn5_higgs = np.array([[31115, 1950], [11022, 5913]])
    precision["RFNN 5 Higgs"] = float(5913) / (5913 + 1950)
    recall["RFNN 5 Higgs"] = float(5913) / (5913 + 11022)
    plot_confusion_matrix(cm_rfnn5_higgs, higgs_labels, title="RFNN5 Higgs")

    cm_rfnn5_cancer = np.array([[54, 11], [2, 33]])
    precision["RFNN 5 Cancer"] = float(33) / (33 + 11)
    recall["RFNN 5 Cancer"] = float(33) / (33 + 2)
    plot_confusion_matrix(cm_rfnn5_cancer, cancer_labels, title="RFNN5 Cancer")

    cm_rfnn10_higgs = np.array([[32495, 570], [14012, 2923]])
    precision["RFNN 10 Higgs"] = float(2923) / (2923 + 570)
    recall["RFNN 10 Higgs"] = float(2923) / (2923 + 14012)
    plot_confusion_matrix(cm_rfnn10_higgs, higgs_labels, title="RFNN10 Higgs")

    cm_rfnn10_cancer = np.array([[51, 14], [0, 35]])
    precision["RFNN 10 Cancer"] = float(35) / (35 + 14)
    recall["RFNN 10 Cancer"] = float(35) / (35 + 0)
    plot_confusion_matrix(cm_rfnn10_cancer, cancer_labels, title="RFNN10s Cancer")

    cm_rfnn30_higgs = np.array([[32652, 413], [14406, 2529]])
    plot_confusion_matrix(cm_rfnn30_higgs, higgs_labels, title="RFNN30 Higgs")

    cm_rfnn30_cancer = np.array([[44, 21], [0, 35]])
    plot_confusion_matrix(cm_rfnn30_cancer, cancer_labels, title="RFNN30 Cancer")

    cm_rfnn50_higgs = np.array([[32846, 219], [15383, 1552]])
    precision["RFNN 50 Higgs"] = float(1552) / (1552 + 219)
    recall["RFNN 50 Higgs"] = float(1552) / (1552 + 15383)
    plot_confusion_matrix(cm_rfnn50_higgs, higgs_labels, title="RFNN50 Higgs")

    cm_rfnn50_cancer = np.array([[48, 17], [0, 35]])
    precision["RFNN 50 Cancer"] = float(35) / (35 + 17)
    recall["RFNN 50 Cancer"] = float(35) / (35 + 0)
    plot_confusion_matrix(cm_rfnn50_cancer, cancer_labels, title="RFNN50 Cancer")

    # DtreeNN
    cm_dtreenn5_higgs = np.array([[22615, 10450], [6541, 10394]])
    precision["DTNN 5 Higgs"] = float(10394) / (10394 + 10450)
    recall["DTNN 5 Higgs"] = float(10394) / (10394 + 6541)
    plot_confusion_matrix(cm_dtreenn5_higgs, higgs_labels, title="DtreeNN5 Higgs")

    cm_dtreenn5_cancer = np.array([[50, 15], [1, 34]])
    precision["DTNN 5 Cancer"] = float(34) / (34 + 15)
    recall["DTNN 5 Cancer"] = float(34) / (34 + 1)
    plot_confusion_matrix(cm_dtreenn5_cancer, cancer_labels, title="DtreeNN5 Cancer")

    cm_dtreenn10_higgs = np.array([[33047, 18], [16694, 241]])
    precision["DTNN 10 Higgs"] = float(241) / (241 + 18)
    recall["DTNN 10 Higgs"] = float(241) / (241 + 16694)
    plot_confusion_matrix(cm_dtreenn10_higgs, higgs_labels, title="DtreeNN10 Higgs")

    cm_dtreenn10_cancer = np.array([[43, 22], [0, 35]])
    precision["DTNN 10 Cancer"] = float(35) / (35 + 22)
    recall["DTNN 10 Cancer"] = float(35) / (35 + 0)
    plot_confusion_matrix(cm_dtreenn10_cancer, cancer_labels, title="DtreeNN10 Cancer")

    cm_dtreenn30_higgs = np.array([[27734, 5331], [11564, 5371]])
    plot_confusion_matrix(cm_dtreenn30_higgs, higgs_labels, title="DtreeNN30 Higgs")

    cm_dtreenn30_cancer = np.array([[56, 9], [1, 34]])
    precision["DTNN 30 Cancer"] = float(34) / (34 + 9)
    recall["DTNN 30 Cancer"] = float(34) / (34 + 1)
    plot_confusion_matrix(cm_dtreenn30_cancer, cancer_labels, title="DtreeNN30 Cancer")

    cm_dtreenn50_higgs = np.array([[24535, 8530], [7828, 9107]])
    precision["DTNN 50 Higgs"] = float(9107) / (9107 + 8530)
    recall["DTNN 50 Higgs"] = float(9107) / (9107 + 7828)
    plot_confusion_matrix(cm_dtreenn50_higgs, higgs_labels, title="DtreeNN50 Higgs")

    cm_dtreenn50_cancer = np.array([[53, 12], [1, 34]])
    precision["DTNN 50 Cancer"] = float(34) / (34 + 12)
    recall["DTNN 50 Cancer"] = float(34) / (34 + 1)
    plot_confusion_matrix(cm_dtreenn50_cancer, cancer_labels, title="DtreeNN50 Cancer")

    print("Precision:\n")
    for k,v in sorted(precision.items()):
        print("%s: %f" % (k, v))

    print("\n\nRecall:\n")
    for k,v in sorted(recall.items()):
        print("%s: %f" % (k, v))


"""
Generates a confusion matrix for the given data
"""
def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=36)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__=="__main__":
    main()
