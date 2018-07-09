from math import sqrt

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score, \
    roc_curve, auc, fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np


class DataTools:

    # 单个模型的结果list
    @staticmethod
    def compute_score_list(y_true, y_pred, y_pred_prob, cost_time):
        result = {}
        recall_, accuracy_, precision_, f1_, f5_, auc_, g_mean_, fpr_, tpr_ = \
            DataTools.compute_score(y_true, y_pred, y_pred_prob)
        result['recall'] = recall_
        result['acc'] = accuracy_
        result['precision'] = precision_
        result['f1'] = f1_
        result['f5'] = f5_
        result['auc'] = auc_
        result['gmean'] = g_mean_
        result['fpr'] = fpr_
        result['tpr'] = tpr_
        result['time'] = cost_time
        return result

    @staticmethod
    def compute_score(y_true, y_pred, y_pred_prob):
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        f5 = fbeta_score(y_true, y_pred, beta=5)
        auc1 = roc_auc_score(y_true, y_pred_prob[:, 1])

        print('*************************************************')
        print('auc:', auc1)
        print('recall:', recall)
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('f1:', f1)
        print('f5', f5)
        # 计算G-mean值
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print('specificity:', specificity)
        g_mean = sqrt(specificity * recall)
        print('g_mean:', g_mean)
        # roc曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        print('auc:', roc_auc)
        print('*******************************************************')
        return recall, accuracy, precision, f1, f5, auc1, g_mean, fpr, tpr

    @staticmethod
    def print_data_ratio(y):
        normal_trans_perc = sum(y['Class'] == 0) / (sum(y['Class'] == 0) + sum(y['Class'] == 1))
        fraud_trans_perc = 1 - normal_trans_perc
        print('Total number : {} '.format(len(y)))
        print('Total number of normal transactions : {}'.format(sum(y['Class'] == 0)))
        print('Total number of  fraudulent transactions : {}'.format(sum(y['Class'] == 1)))
        print('Percent of normal transactions is : {:.4f}%,  '
              'fraudulent transactions is : {:.4f}%'.format(normal_trans_perc * 100, fraud_trans_perc * 100))

    @staticmethod
    def data_split(X, y, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print('数据集总数：', len(X_train) + len(X_test))
        print('分割后的训练集：', len(X_train))
        print('分割后的测试集：', len(X_test))
        print('\n')
        return X_train, X_test, y_train, y_test

    @staticmethod
    # 计算混淆矩阵
    def compute_confusion_matrix(y_true, y_pred):
        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        # print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
        return cnf_matrix
