from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from tools.tools_data import DataTools


class ModelAll:

    @staticmethod
    def grid_search_cv_all(X_train, y_train):
        print('开始交叉验证。。。')
        start = time.time()
        param_grid = {
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
            'C': [0.1, 1, 10, 100]
        }
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
        CV_cfl = GridSearchCV(estimator=XGBClassifier(),
                              param_grid=param_grid,
                              cv=kfold,
                              scoring='f1',
                              verbose=1,
                              n_jobs=-1)
        CV_cfl.fit(X_train, y_train.values.ravel())
        print('最佳参数是：', CV_cfl.best_params_)
        end = time.time()
        print('cv耗时：', end - start)


    @staticmethod
    def model_all_own(clf, X_train, y_train, X_test, y_test):

        print('*******************************************')
        print(clf.__class__.__name__, '开始fit...')
        start_time = time()
        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        y_perd_prob = clf.predict_proba(X_test)
        end_time = time()

        result = {}
        roc_pr = {}
        recall_, accuracy_, precision_, f1_, f5_, auc_, g_mean_, fpr_, tpr_ = \
            DataTools.compute_score(y_test, y_pred, y_perd_prob)

        result['recall'] = recall_
        result['acc'] = accuracy_
        result['precision'] = precision_
        result['f1'] = f1_
        result['f5'] = f5_
        result['auc'] = auc_
        result['gmean'] = g_mean_
        result['time'] = end_time - start_time

        roc_pr['fpr'] = fpr_
        roc_pr['tpr'] = tpr_

        print("{} 训练结束，耗时： {:.4f} ".format(clf.__class__.__name__, (end_time - start_time)))

        return result, roc_pr
