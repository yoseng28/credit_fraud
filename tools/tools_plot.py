import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tools.tools_data import DataTools


class PlotTools:

    @staticmethod
    def plot_xgb_roc_curve(file_path):
        data = pd.read_excel(file_path)
        clfs = ['XGB', 'XGB_Tuned', 'XGB+EE', 'XGB+EE+Smote', 'XGB+EE+BSO1',  'XGB+EE+BSO2']
        for i in range(0, 6):
            fpr = data.iloc[:, [i*2]].dropna(axis=0)
            tpr = data.iloc[:, [i*2+1]].dropna(axis=0)
            # 绘制ROC曲线图
            PlotTools.plot_roc_curve2(fpr, tpr, clfs[i])

    # 绘制精度-召回率曲线
    @staticmethod
    def plot_precision_recall(y_true, y_pred_proba):
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        colors = itertools.cycle(
            ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue', 'black'])
        plt.figure(figsize=(5, 5))
        for i, color in zip(thresholds, colors):
            y_predictions_prob = y_pred_proba[:, 1] > i
            precision, recall, thresholds = precision_recall_curve(y_true, y_predictions_prob)
            plt.plot(recall, precision, color=color, label='Threshold: %s' % i)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall example')
            plt.legend(loc="lower left")

    # 绘制阈值图
    @staticmethod
    def plot_thresholds(y_true, y_pred_proba):
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        plt.figure(figsize=(10, 10))
        j = 1
        for i in thresholds:
            y_predictions_high_recall = y_pred_proba[:, 1] > i
            plt.subplot(3, 3, j)
            j += 1
            cnf_matrix = DataTools.compute_confusion_matrix(y_true, y_predictions_high_recall)
            class_names = [0, 1]
            PlotTools.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)


    @staticmethod
    def plot_precision_recall_curve(recall, precision, precision_score, clf_name):
        plt.figure()
        plt.plot(recall, precision, label='%s Precision (area = %0.2f)' % (clf_name, precision_score))
        plt.legend(loc=3, fontsize=8)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    # 绘制ROC曲线2
    @staticmethod
    def plot_roc_curve2(fpr, tpr, clf_name):
        roc_auc = auc(fpr, tpr)
        # plt.title('Receiver Operating Characteristic')
        plt.figure(1)
        plt.plot(fpr, tpr, label='%s  (AUC = %0.5f)' % (clf_name, roc_auc))
        plt.legend(loc=4, fontsize=8)
        plt.xlim([-0.01, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    # 绘制ROC曲线
    @staticmethod
    def plot_roc_curve(y_true, y_score):
        plt.figure()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
        # plt.legend(loc='lower right')
        plt.legend(loc=4, fontsize=8)
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 0.51])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # 绘制混淆矩阵
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, classes=[0, 1], normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        if normalize:
            # print("Normalized confusion matrix")
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        else:
            # print('Confusion matrix, without normalization')
            1
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    # 数据直方图
    @staticmethod
    def fraud_class_histogram(data):
        # 计算一列不同的值
        count_class = pd.value_counts(data['Class']).sort_index()
        print('类别不同数量：/n%s' % count_class)
        count_class.plot(kind='bar', colormap='Oranges_r')
        plt.title('Fraud Class Histogram')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
