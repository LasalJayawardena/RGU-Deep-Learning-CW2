from itertools import cycle
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import random


def plot_multi_class_roc_auc(y_test, y_pred, class_labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']):
    """
    Plot Roc Curves for a Multi-class Classification Task
    """

    plt.style.use(['default'])
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
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix
    """
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (20, 10))
    cm = pd.DataFrame(cm , index = [i for i in labels] , columns = [i for i in labels])
    sns.set(font_scale=1)
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Fashion Mnist Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.show()

def visualize_data_subset(data, target, title="Train Data Visualization", num_images=25, display_labels=True):
    """
    Visualize a set of Images
    """
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Set the figure size
    fig = plt.figure(figsize=(10,10))
    # Show only the first 50 pictures
    plt.suptitle(title, fontsize=20)
    # Inex range
    index_l = random.sample(range(0, len(target)), num_images+1)
    for i in range(num_images):
        plt.subplot(5,num_images//5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data[index_l[i]], cmap=plt.cm.binary)
        if display_labels:
            plt.xlabel(labels[target[index_l[i]]])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_history(history):
    """
    Plot history after model training
    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_single_image(image):
    fig = plt.figure(figsize=(2, 2))
    plt.imshow(image,  aspect='equal')
    plt.grid(b=None)
    plt.axis("off");