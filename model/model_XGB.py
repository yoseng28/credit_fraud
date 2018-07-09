import time

from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb


class ModelXGB:

    @staticmethod
    def xgb_predict_prob(X_train, y_train, X_test):
        xgb_clf = XGBClassifier(
            learning_rate=0.4,
            n_estimators=170,
            max_depth=2,
            min_child_weight=1,
            gamma=0,
            objective='binary:logistic',
            subsample=0.2,
            colsample_bytree=0.1,
            reg_lambda=0.1
        )
        print('开始fit(prob)......')
        fit_start_time = time.clock()
        xgb_clf.fit(X_train, y_train.values.ravel())
        y_predict_score = xgb_clf.predict_proba(X_test)
        fit_end_time = time.clock()
        print('fit耗时(prob)：', fit_end_time - fit_start_time)
        return y_predict_score

    @staticmethod
    def xgb_predict(X_train, y_train, X_test):
        xgb_clf = XGBClassifier(
            learning_rate=0.2,
            n_estimators=70,
            max_depth=4,
            min_child_weight=1,
            gamma=0,
            objective='binary:logistic',
            subsample=0.6,
            colsample_bytree=0.4,
            reg_lambda=0.1
        )
        print('开始fit......')
        fit_start_time = time.clock()
        xgb_clf.fit(X_train, y_train.values.ravel())
        y_predict = xgb_clf.predict(X_test)
        fit_end_time = time.clock()
        print('fit耗时：', fit_end_time - fit_start_time)
        return y_predict

    @staticmethod
    def xgb_gridSearchCV(X_train, y_train):
        # 暂时调整的参数
        max_depth = range(4, 7, 1)
        min_child_weight = range(1, 3, 1)
        # param_grid = dict(max_depth=max_depth, min_child_weight=min_child_weight)
        # subsample = [i / 10.0 for i in range(0, 10)]
        # colsample_bytree = [i / 10.0 for i in range(1, 10)]
        # reg_lambda = [i / 10.0 for i in range(1, 10)]
        param_grid = dict(max_depth=max_depth, min_child_weight=min_child_weight)
        learning_rate = [i / 10.0 for i in range(1, 10)]
        # param_grid = dict(learning_rate=learning_rate)
        xgb_beta = XGBClassifier(
            learning_rate=0.1,
            n_estimators=70,
            # max_depth=4,
            # min_child_weight=1,
            gamma=0,
            objective='binary:logistic',
            # subsample=0.6,
            # colsample_bytree=0.4,
            reg_lambda=0.1)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
        # n_jobs = -1 并行处理交叉验证
        print('开始cv......')
        cv_start_time = time.clock()
        grid_search = GridSearchCV(xgb_beta, param_grid=param_grid, scoring='f1', n_jobs=-1,
                                   cv=kfold, return_train_score=True)
        cv_end_time = time.clock()
        time_cv_cost = (cv_end_time - cv_start_time)
        print('cv耗时：', time_cv_cost)
        print('开始fit......')
        fit_start_time = time.clock()
        grid_search.fit(X_train, y_train.values.ravel())
        fit_end_time = time.clock()
        time_fit_cost = (fit_end_time - fit_start_time)
        print('fit耗时：', time_fit_cost)
        print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        DataFrame(grid_search.cv_results_).to_csv('data/result/xgb_ee_GridSearchCV_rate.csv')

    # cv1:n_estimators参数
    @staticmethod
    def xgb_cv_param(X_train, y_train, early_stopping_rounds=50):
        cv_param = 'n_estimators'
        # cv_param = 'gamma'
        DTrain = xgb.DMatrix(X_train.values, label=y_train.values.ravel())

        # StratifiedKFold：采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
        SKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

        xgb_beta = XGBClassifier(
            learning_rate=0.1,
            n_estimators=70,
            max_depth=6,
            min_child_weight=2,
            # gamma=0,
            # subsample=0.6,
            # colsample_bytree=0.4,
            objective='multi:softmax',
            # reg_lambda=0.1
        )

        xgb_param = xgb_beta.get_xgb_params()
        xgb_param['num_class'] = 2
        # 交叉验证
        print('进行交叉验证......')
        time_cv_start = time.clock()
        cv_result = xgb.cv(xgb_param, DTrain, num_boost_round=xgb_param[cv_param], folds=SKFold,
                           metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        print('交叉验证结束！')
        print('参数停止数为：', cv_result.shape[0])
        time_cv_end = time.clock()
        time_cv_cost = (time_cv_end - time_cv_start)
        print('耗时：', time_cv_cost)
        # print('cv_result:\n', cv_result)
        cv_result.to_csv('data/result/ee_smote_cv_n_estimators_result.csv', index_label='n_estimators')
        print('文件生成成功！')
