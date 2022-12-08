from itertools import cycle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



def plot_multi_class_roc_auc(y_test, y_pred, class_labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']):

    
    if class_labels is None:
        num_classes = len(np.unique(y_test))
        class_labels = list(range(num_classes))
    else:
        num_classes = len(class_labels)

    y_test = label_binarize(y_test, classes=np.arange(num_classes))
    y_pred = label_binarize(y_pred, classes=np.arange(num_classes))

    # Compute ROC curve and ROC area for each class
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for i in range(num_classes):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])
    
    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    # First aggregate all false positive rates
    all_false_positive_rate = np.unique(np.concatenate([false_positive_rate[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_true_positive_rate = np.zeros_like(all_false_positive_rate)
    for i in range(num_classes):
        mean_true_positive_rate += np.interp(all_false_positive_rate, false_positive_rate[i], true_positive_rate[i])

    # Finally average it and compute AUC
    mean_true_positive_rate /= num_classes

    false_positive_rate["macro"] = all_false_positive_rate
    true_positive_rate["macro"] = mean_true_positive_rate
    roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])

    # Plot all ROC curves
    #plt.figure(figsize=(10,5))
    plt.figure(dpi=600)
    lw = 1
    plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
        label="Micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink", linestyle=":", linewidth=1.5,)

    plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
        label="Macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy", linestyle=":", linewidth=1.5,)

    colors = cycle(mcolors.TABLEAU_COLORS.keys())
    for i, color in zip(range(num_classes), colors):
        plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(class_labels[i], roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curve")
    plt.rcParams.update({'font.size': 6})
    plt.legend()