import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_preds, y_true, classes):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_preds,
        display_labels=classes,
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax1,
    )
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_preds,
        display_labels=classes,
        cmap=plt.cm.Blues,
        ax=ax2,
    )
    plt.show()

def history_plot(history):
    x = range(1, len(history) + 1)
    plt.plot(x, history)
    plt.show()

def history_plot_adv(*args):
    x = range(1, len(args[0]) + 1)
    for y in args:
        plt.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("validation accuracy")
    plt.show()

def print_fold_iteration_score(score, prefix = ""):
    for iter in score:
        for k in list(iter.keys()):
            val = format(iter[k], '.3f')
            print(f'{prefix}{k} - {val}', end = ' | ') 
        print("")

def print_score(score, prefix = ""):
    for k in list(score.keys()):
        val = format(score[k], '.3f')
        print(f'{prefix}{k} - {val}', end = ' | ') 
    print("")

def print_score_tot(score_tot):
  for key in list(score_tot.keys()):
    print(key)
    print_score(score_tot[key])


def ECGplot(ecg, title="", transposed=True):
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    x = []
    if transposed:
        x = np.arange(ecg.shape[0])
    else:
        x = np.arange(ecg.shape[1])

    fig, ax = plt.subplots(6, 2)
    
    fig.suptitle(title)
    
    j = 0
    k = 0

    for i in range(0, 12):
        if i == 6:
            j = 1
        k = i % 6
        if transposed:
            ax[k, j].plot(x, ecg[:, i], linewidth=2.0)
        else:
            ax[k, j].plot(x, ecg[i, :], linewidth=2.0)
        ax[k, j].set_title(leads[i], loc="left")
    
    plt.show()

def ECGplot_adv(data1, data2, title="", transposed=True):
    leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    x = []
    if transposed:
        x = np.arange(data1.shape[0])
    else:
        x = np.arange(data1.shape[1])

    fig, ax = plt.subplots(6, 2)
    
    fig.suptitle(title)
    
    j = 0
    k = 0

    for i in range(0, 12):
        if i == 6:
            j = 1
        k = i % 6
        if transposed:
            ax[k, j].plot(x, data1[:, i], linewidth=2.0)
            ax[k, j].plot(x, data2[:, i], linewidth=2.0)
        else:
            ax[k, j].plot(x, data1[i, :], linewidth=2.0)
            ax[k, j].plot(x, data2[i, :], linewidth=2.0)
        ax[k, j].set_title(leads[i], loc="left")
    
    plt.show()
