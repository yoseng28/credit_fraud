import time

from imbalance.smote_EE import SmoteEE
from imbalance.under_sample import UnderSample
from model.model_LR import ClassifierLR
from preprocessing.preprocessing_data import DataPreprocessing
from tools.tools_data import DataTools
from tools.tools_plot import PlotTools

import pandas as pd
import numpy as np


class ResultLR:

    @staticmethod
    def LR_EE_smote(data):
        # 子集数目
        num_subsets = 10
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)

        result = {}
        result_recall = []
        result_acc = []
        result_precision = []
        result_f1 = []
        result_auc = []
        result_gmean = []
        result_fpr_temps = []
        result_tpr_temps = []

        start_time = time.clock()
        for i in (range(num_subsets)):
            print('******************************************************************************')
            print('第 ', i + 1, ' 个分类器开始：')
            # EE&smote后的数据

            X_ee_smote, y_ee_smote = SmoteEE.smoteEE_own(X_train, y_train)
            DataTools.print_data_ratio(y_ee_smote)

            # 训练参数
            # print('训练集子集%d：' % (i + 1))
            # ClassifierLR.lr_grid_search_cv(X_ee_smote, y_ee_smote)

            pd.concat([X_ee_smote, y_ee_smote], axis=1).to_csv('data/subsets/lr_subset%d.csv' % (i + 1))
            print('第%d个 子集导出成功！' % (i + 1))
            print('训练集子集%d：' % (i + 1))
            DataTools.print_data_ratio(y_ee_smote)

            y_predict = ClassifierLR.fit_model_LR(X_ee_smote, y_ee_smote, X_test, 0.1)
            y_predict_prob = ClassifierLR.lr_predict_proba(X_ee_smote, y_ee_smote, X_test, 0.1)
            recall_, accuracy_, precision_, f1_, auc_, g_mean_, fpr_, tpr_ = \
                DataTools.compute_score(y_test, y_predict, y_predict_prob)
            result_recall.append(recall_)
            result_acc.append(accuracy_)
            result_precision.append(precision_)
            result_f1.append(f1_)
            result_auc.append(auc_)
            result_gmean.append(g_mean_)
            result_fpr_temps.append(fpr_)
            result_tpr_temps.append(tpr_)

        end_time = time.clock()
        result['time'] = end_time - start_time
        result['recall'] = np.mean(result_recall)
        result['acc'] = np.mean(result_acc)
        result['precision'] = np.mean(result_precision)
        result['f1'] = np.mean(result_f1)
        result['auc'] = np.mean(result_auc)
        result['gmean'] = np.mean(result_gmean)
        result['fpr'] = pd.DataFrame(result_fpr_temps).mean()
        result['tpr'] = pd.DataFrame(result_tpr_temps).mean()
        pd.DataFrame(result).to_csv('data/score/lr_ee_tuned.csv')
        print('结果已保存至score文件夹下 ^_^')

        # plt.figure(1)
        # plt.plot(result['fpr'], result['tpr'], label='%s ROC (area = %0.2f)' % (i,  result['auc']))
        # plt.legend(loc=4, fontsize=8)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])


    # LR+原始数据
    # 采用predict去fit，生成预测标签
    @staticmethod
    def LR_origin_data(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        # 训练LR，得到C参数
        c_param_scores = ClassifierLR.c_param_scores(X_train, y_train)
        # 训练模型
        y_predict = ClassifierLR.fit_model_LR(X_train, y_train, X_test, c_param_scores)
        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')
        # 绘制ROC曲线
        ResultLR.LR_plot_ROC(X_train, y_train, X_test, y_test, c_param_scores)
        # 绘制阈值图
        # ResultLR.LR_plot_threshold(X_train, y_train, X_test, y_test, c_param_scores)

    # LR+下采样测试数据
    # 采用predict去fit，生成预测标签
    @staticmethod
    def LR_under_sample(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        # 下采样后的新数据
        X_under_sample, y_under_sample = UnderSample.under_sample_own(X_train, y_train)
        X_under_sample_train, X_under_sample_test, y_under_sample_train, y_under_sample_test = DataTools.data_split(
            X_under_sample, y_under_sample)
        # 训练LR，得到C参数
        c_param_scores = ClassifierLR.c_param_scores(X_under_sample_train, y_under_sample_train)
        # 训练模型
        y_predict = ClassifierLR.fit_model_LR(X_under_sample_train, y_under_sample_train, X_test, c_param_scores)
        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')
        # 绘制ROC曲线
        ResultLR.LR_plot_ROC(X_under_sample_train, y_under_sample_train, X_test, y_test, c_param_scores)
        # 绘制阈值图
        # ResultLR.LR_plot_threshold(X_under_sample_train, y_under_sample_train, X_test,
        #                            y_test, c_param_scores)

    # LR+下采样的训练数据
    # 采用predict去fit，生成预测标签
    @staticmethod
    def LR_under_sample_test(data):
        # 下采样后的新数据
        X_under_sample, y_under_sample = UnderSample.under_sample_own(data)
        X_under_sample_train, X_under_sample_test, y_under_sample_train, y_under_sample_test = DataTools.data_split(
            X_under_sample, y_under_sample)
        # 训练LR，得到C参数
        c_param_scores = ClassifierLR.c_param_scores(X_under_sample_train, y_under_sample_train)
        # 训练模型
        y_predict = ClassifierLR.fit_model_LR(X_under_sample_train, y_under_sample_train, X_under_sample_test,
                                              c_param_scores)
        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_under_sample_test, y_predict)
        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')
        # 绘制ROC曲线
        ResultLR.LR_plot_ROC(X_under_sample_train, y_under_sample_train, X_under_sample_test, y_under_sample_test,
                             c_param_scores)
        # 绘制阈值图
        # ResultLR.LR_plot_threshold(X_under_sample_train, y_under_sample_train, X_under_sample_test,
        #                            y_under_sample_test, c_param_scores)

        # 绘制精度-召回率曲线
        ResultLR.LR_plot_precision_recall(X_under_sample_train, y_under_sample_train, X_under_sample_test,
                                          y_under_sample_test, c_param_scores)

    # 绘制阈值图
    @staticmethod
    def LR_plot_threshold(X_train, y_train, X_test, y_test, c_param_scores):
        y_pred_proba = ClassifierLR.lr_predict_proba(X_train, y_train, X_test, c_param_scores)
        PlotTools.plot_thresholds(y_test, y_pred_proba)

    # 绘制ROC曲线
    @staticmethod
    def LR_plot_ROC(X_train, y_train, X_test, y_test, c_param_scores):
        y_predict_score = ClassifierLR.lr_roc_curve_score(X_train, y_train, X_test, c_param_scores)
        PlotTools.plot_roc_curve(y_test, y_predict_score)

    # 绘制精度-召回率曲线
    @staticmethod
    def LR_plot_precision_recall(X_train, y_train, X_test, y_test, c_param_scores):
        y_pred_proba = ClassifierLR.lr_predict_proba(X_train, y_train, X_test, c_param_scores)
        PlotTools.plot_precision_recall(y_test, y_pred_proba)
