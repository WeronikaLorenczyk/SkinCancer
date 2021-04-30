import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(Y_true,Y_pred_classes, data_type, save_location=None, title='Confusion matrix'):
    cm = confusion_matrix(Y_true, Y_pred_classes)

    if data_type == "all":
        classes = ('bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec')

    else:
        classes = ('mel', 'nv')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()
    plt.clf()


def plot_roc(test_y, Y_pred, save_location=None):
    fpr, tpr, _ = roc_curve(test_y, Y_pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.subplots_adjust(0.12,0.1,0.9,0.87)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()
    plt.clf()
