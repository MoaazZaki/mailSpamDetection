import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


def get_accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


def analyze_result(y_pred, y_test):
    print("Accuary: {:.2f}%".format(100 * get_accuracy(y_pred, y_test)))
    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
    print("F1 Score: {:.2f}%".format(100 * f1_score(y_test, y_pred)))


def plot_cunfusion_matrix(y_pred, y_test):
    cf_matrix = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='YlGn', fmt='')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Spam', 'Spam'])
    ax.yaxis.set_ticklabels(['Not Spam', 'Spam'])
