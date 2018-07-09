# 逻辑回归分类器
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd

# 正则项惩罚力度，C值越小，则正则化强度越大
# 0.01为最佳
C_PARAM_RANGE = [0.01, 0.1, 1, 10, 100]
N_SPLITS = 5
# L1正则
PENALTY = 'l1'


class ClassifierLR:

    @staticmethod
    def lr_predict_proba(X_train, y_train, X_test, c_param_scores):
        lr = LogisticRegression(C=c_param_scores, penalty=PENALTY)
        lr.fit(X_train, y_train.values.ravel())
        y_pred_proba = lr.predict_proba(X_test.values)
        return y_pred_proba

    @staticmethod
    def fit_model_LR(X_train, y_train, X_test, c_param_scores):
        lr = LogisticRegression(C=c_param_scores, penalty='l1', random_state=0)
        lr.fit(X_train, y_train.values.ravel())
        y_predict = lr.predict(X_test.values)
        return y_predict

    @staticmethod
    def lr_grid_search_cv(X_train, y_train):
        print('开始交叉验证。。。')
        start = time.time()
        param_grid = {
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
            'C': [0.1, 1, 10, 100]
        }
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
        CV_cfl = GridSearchCV(estimator=LogisticRegression(),
                              param_grid=param_grid,
                              cv=kfold,
                              scoring='recall',
                              verbose=1,
                              n_jobs=-1)
        CV_cfl.fit(X_train, y_train.values.ravel())
        best_parameters = CV_cfl.best_params_
        print('最佳参数是：', best_parameters)
        end = time.time()
        print('cv耗时：', end - start)

    # ROC曲线
    @staticmethod
    def lr_roc_curve_score(X_train, y_train, X_test, c_param_scores):
        lr = LogisticRegression(C=c_param_scores, penalty=PENALTY)
        y_predict_score = lr.fit(X_train, y_train.values.ravel()).decision_function(X_test.values)
        return y_predict_score

    # 训练C参数
    @staticmethod
    def c_param_scores(X_train_data, y_train_data):

        fold = KFold(N_SPLITS, shuffle=False)
        results_table = pd.DataFrame(index=range(len(C_PARAM_RANGE), 2), columns=['C_parameter', 'recall_mean'])
        results_table['C_parameter'] = C_PARAM_RANGE
        print('results_table:', results_table)

        j = 0
        for c_param in C_PARAM_RANGE:
            print('-------------------------------------------')
            print('C parameter: ', c_param)
            print('-------------------------------------------')
            print('')
            recall_accs = []
            i = 1
            for train_index, test_index in fold.split(X_train_data):
                lr = LogisticRegression(C=c_param, penalty=PENALTY)
                lr.fit(X_train_data.iloc[train_index, :], y_train_data.iloc[train_index, :].values.ravel())
                y_pred_undersample = lr.predict(X_train_data.iloc[test_index, :].values)
                recall_acc = recall_score(y_train_data.iloc[test_index, :].values, y_pred_undersample)
                recall_accs.append(recall_acc)
                print('迭代次数 ', i, ': recall score = ', recall_acc)
                i += 1

            results_table.ix[j, 'recall_mean'] = np.mean(recall_accs)
            print('results_table:', results_table)
            j += 1
            print('')
            print('recall平均值： ', np.mean(recall_accs))
            print('')
        print('results_table:', results_table)
        best_recall = results_table.loc[results_table['recall_mean'].idxmax()]['recall_mean']
        best_c = results_table.loc[results_table['recall_mean'].idxmax()]['C_parameter']

        print('*********************************************************************************')
        print('最好的模型 recall = ', best_recall)
        print('最好的模型 C parameter = ', best_c)
        print('*********************************************************************************')
        return best_c
