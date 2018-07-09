import time

from sklearn.metrics import recall_score

from imbalance.EE import EE
from imbalance.over_sample import OverSample
from imbalance.smote_EE import SmoteEE
from imbalance.smote_origin import SmoteOrigin
from imbalance.under_sample import UnderSample
from model.model_XGB import ModelXGB
from preprocessing.preprocessing_data import DataPreprocessing
from tools.tools_data import DataTools
from tools.tools_plot import PlotTools

import numpy as np
import pandas as pd


class ResultXGB:

    @staticmethod
    def XGB_origin(data):
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train_temp, X_test, y_train_temp, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集temp：')
        DataTools.print_data_ratio(y_train_temp)

        X_train, X_validate, y_train, y_validate = DataTools.data_split(X_train_temp, y_train_temp)
        print('分割后的验证集：')
        DataTools.print_data_ratio(y_validate)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)

        # sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=666)
        # sss.get_n_splits(X, y)

        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_train, y_train, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_train, y_train, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score2/xgb_origin.csv')
        print('结果已保存至score文件夹下 ^_^')

        # # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

        PlotTools.plot_roc_curve(y_test, y_predict_prob[:, 1])

    @staticmethod
    def XGB_EE(data):
        # 子集数目
        num_subsets = 5
        X, y = DataPreprocessing.read_X_y(data)
        X_train_tmp, X_test, y_train_tmp, y_test = DataTools.data_split(X, y)
        X_train, X_validate, y_train, y_validate = DataTools.data_split(X_train_tmp, y_train_tmp)

        result = {}
        result_recall = []
        result_acc = []
        result_precision = []
        result_f1 = []
        result_f5 = []
        result_auc = []
        result_gmean = []
        result_fpr_temps = []
        result_tpr_temps = []

        start_time = time.clock()
        for i in (range(num_subsets)):
            print('******************************************************************************')
            print('第 ', i + 1, ' 个分类器开始：')
            # EE&smote后的数据
            X_ee, y_ee = EE.ee_own(X_train, y_train)
            pd.concat([X_ee, y_ee], axis=1).to_csv('data/subsets/subset_ee%d.csv' % (i + 1))
            print('第%d个 子集导出成功！' % (i + 1))
            print('训练集子集%d：' % (i + 1))
            DataTools.print_data_ratio(y_ee)

            # 训练参数
            # ModelXGB.xgb_cv_param(X_ee, y_ee)
            # ModelXGB.xgb_gridSearchCV(X_ee, y_ee)
            # return

            y_predict = ModelXGB.xgb_predict(X_ee, y_ee, X_test)
            y_predict_prob = ModelXGB.xgb_predict_prob(X_ee, y_ee, X_test)
            recall_, accuracy_, precision_, f1_, f5_, auc_, g_mean_, fpr_, tpr_ = \
                DataTools.compute_score(y_test, y_predict, y_predict_prob)
            result_recall.append(recall_)
            result_acc.append(accuracy_)
            result_precision.append(precision_)
            result_f1.append(f1_)
            result_f5.append(f5_)
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
        result['f5'] = np.mean(result_f5)
        result['auc'] = np.mean(result_auc)
        result['gmean'] = np.mean(result_gmean)
        result['fpr'] = pd.DataFrame(result_fpr_temps).mean()
        result['tpr'] = pd.DataFrame(result_tpr_temps).mean()
        pd.DataFrame(result).to_csv('data/score2/xgb_ee3.csv')
        print('结果已保存至score文件夹下 ^_^')
        # # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

        PlotTools.plot_roc_curve(y_test, y_predict_prob[:, 1])

    @staticmethod
    def XGB_under_sample_smote(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)

        # 下采样后的新数据
        X_under_sample_train, y_under_sample_train = UnderSample.under_sample_own(X_train, y_train)
        print('下采样后的训练集：')
        DataTools.print_data_ratio(y_under_sample_train)

        # smote后的数据
        X_smote, y_smote = SmoteOrigin.smote_own(X_under_sample_train, y_under_sample_train)
        print('SMOTE后的训练集：')
        DataTools.print_data_ratio(y_smote)

        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_smote, y_smote, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_smote, y_smote, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score2/aaa/ee_smote2.csv')
        print('结果已保存至score文件夹下 ^_^')
        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')
        PlotTools.plot_roc_curve(y_test, y_predict_prob[:, 1])
        print('**************************************************************************')

    @staticmethod
    def XGB_smote(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)
        # smote后的数据
        X_smote, y_smote = SmoteOrigin.smote_own(X_train, y_train)
        print('SMOTE后的训练集：')
        DataTools.print_data_ratio(y_smote)

        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_smote, y_smote, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_smote, y_smote, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score/xgb_smote.csv')
        print('结果已保存至score文件夹下 ^_^')

        # # 计算混淆矩阵
        # cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # # 绘制混淆矩阵图
        # PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

    @staticmethod
    def XGB_EE_smote(data):
        # 子集数目
        num_subsets = 1
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train_temp, X_test, y_train_temp, y_test = DataTools.data_split(X, y)
        # print('分割后的测试集：')
        # DataTools.print_data_ratio(y_test)
        # print('分割后的训练集：')
        # DataTools.print_data_ratio(y_train)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集temp：')
        DataTools.print_data_ratio(y_train_temp)

        X_train, X_validate, y_train, y_validate = DataTools.data_split(X_train_temp, y_train_temp)

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
        for i in(range(num_subsets)):
            print('******************************************************************************')
            print('第 ', i+1, ' 个分类器开始：')
            # EE&smote后的数据
            X_ee_smote, y_ee_smote = SmoteEE.smoteEE_own(X_train, y_train)

            # 训练参数
            # ModelXGB.xgb_cv_param(X_ee_smote, y_ee_smote)
            # ModelXGB.xgb_gridSearchCV(X_ee_smote, y_ee_smote)

            pd.concat([X_ee_smote, y_ee_smote], axis=1).to_csv('data/subsets/subset%d.csv' % (i+1))
            print('第%d个 子集导出成功！' % (i+1))
            print('训练集子集%d：' % (i+1))
            DataTools.print_data_ratio(y_ee_smote)

            y_predict = ModelXGB.xgb_predict(X_ee_smote, y_ee_smote, X_test)
            y_predict_prob = ModelXGB.xgb_predict_prob(X_ee_smote, y_ee_smote, X_test)
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
        result['time'] = end_time-start_time
        result['recall'] = np.mean(result_recall)
        result['acc'] = np.mean(result_acc)
        result['precision'] = np.mean(result_precision)
        result['f1'] = np.mean(result_f1)
        result['auc'] = np.mean(result_auc)
        result['gmean'] = np.mean(result_gmean)
        result['fpr'] = pd.DataFrame(result_fpr_temps).mean()
        result['tpr'] = pd.DataFrame(result_tpr_temps).mean()
        pd.DataFrame(result).to_csv('data/score/xgb_smote_reg_ee_tuned.csv')
        print('结果已保存至score文件夹下 ^_^')

        # plt.figure(1)
        # plt.plot(result['fpr'], result['tpr'], label='%s ROC (area = %0.2f)' % (i,  result['auc']))
        # plt.legend(loc=4, fontsize=8)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])

    @staticmethod
    def XGB_smote(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)
        # smote后的数据
        X_smote, y_smote = SmoteOrigin.smote_own(X_train, y_train)
        print('SMOTE后的训练集：')
        DataTools.print_data_ratio(y_smote)

        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_smote, y_smote, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_smote, y_smote, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score/xgb_smote.csv')
        print('结果已保存至score文件夹下 ^_^')

        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)
        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

    # 采用predict去fit，生成预测标签
    @staticmethod
    def XGB_under_sample(data):
        # pandas显示
        # count_class = pd.value_counts(under_sample_data['Class']).sort_index()
        # print('下采用后的class为：', count_class)

        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)
        # 下采样后的新数据
        X_under_sample_train, y_under_sample_train = UnderSample.under_sample_own(X_train, y_train)
        print('下采样后的训练集：')
        DataTools.print_data_ratio(y_under_sample_train)

        # X_under_sample_train, X_under_sample_test, y_under_sample_train, y_under_sample_test = DataTools.data_split(
        #     X_under_sample, y_under_sample)
        # print('下采样分割后的训练集：')
        # DataTools.print_data_ratio(y_under_sample_train)

        # xgb自带cv训练参数
        # ModelXGB.xgb_cv_param(X_under_sample_train, y_under_sample_train)
        # 使用GridSearchCV训练参数
        # ModelXGB.xgb_gridSearchCV(X_under_sample_train, y_under_sample_train)

        # 训练模型
        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_under_sample_train, y_under_sample_train, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_under_sample_train, y_under_sample_train, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score2/xgb_us2.csv')
        print('结果已保存至score文件夹下 ^_^')

        # 计算混淆矩阵
        cnf_matrix = DataTools.compute_confusion_matrix(y_test, y_predict)

        # 绘制混淆矩阵图
        PlotTools.plot_confusion_matrix(cnf_matrix, title='Confusion matrix')
        PlotTools.plot_roc_curve(y_test, y_predict_prob[:, 1])

    @staticmethod
    def XGB_over_sample(data):
        # 采样前的数据
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train, X_test, y_train, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集：')
        DataTools.print_data_ratio(y_train)

        # smote后的数据
        X_os, y_os = OverSample.over_sample_own(X_train, y_train)
        print('上采样后的训练集：')
        DataTools.print_data_ratio(y_os)

        start_time = time.clock()
        y_predict = ModelXGB.xgb_predict(X_os, y_os, X_test)
        y_predict_prob = ModelXGB.xgb_predict_prob(X_os, y_os, X_test)
        end_time = time.clock()
        cost_time = end_time - start_time
        result = DataTools.compute_score_list(y_test, y_predict, y_predict_prob, cost_time)
        pd.DataFrame(result).to_csv('data/score/xgb_oversample.csv')
        print('结果已保存至score文件夹下 ^_^')
