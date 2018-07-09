from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd

from imbalance.EE import EE
from imbalance.over_sample import OverSample
from imbalance.smote_origin import SmoteOrigin
from imbalance.under_sample import UnderSample
from model.model_XGB import ModelXGB
from model.model_all import ModelAll
from preprocessing.preprocessing_data import DataPreprocessing
from tools.tools_data import DataTools
from tools.tools_plot import PlotTools


class ResultAll:

    @staticmethod
    def result_all_model(data):
        X, y = DataPreprocessing.read_X_y(data)
        print('原始数据：')
        DataTools.print_data_ratio(y)
        X_train_temp, X_test, y_train_temp, y_test = DataTools.data_split(X, y)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_test)
        print('分割后的训练集temp：')
        DataTools.print_data_ratio(y_train_temp)

        X_train, X_test_validate, y_train, y_test_validate = DataTools.data_split(X_train_temp, y_train_temp)
        print('分割后的验证集：')
        DataTools.print_data_ratio(y_test_validate)
        print('分割后的测试集：')
        DataTools.print_data_ratio(y_train)

        # X_under_sample_train, y_under_sample_train = SmoteOrigin.smote_own(X_train, y_train)
        # print('smote后的测试集：')
        # DataTools.print_data_ratio(y_under_sample_train)

        # 下采样后的新数据
        # X_under_sample_train, y_under_sample_train = UnderSample.under_sample_own(X_train, y_train)
        # print('下采样后的训练集：')
        # DataTools.print_data_ratio(y_under_sample_train)

        # ModelAll.grid_search_cv_all(X_train, y_train)
        # ModelXGB.xgb_gridSearchCV(X_ee, y_ee)
        # return

        X_under_sample_train, y_under_sample_train = OverSample.over_sample_own(X_train, y_train)
        print('上采样后的测试集：')
        DataTools.print_data_ratio(y_under_sample_train)



        clf_knn = KNeighborsClassifier()

        clf_lr = LogisticRegression()
        clf_dt = DecisionTreeClassifier()
        clf_aba = AdaBoostClassifier()
                        # n_estimators=300,
                        # learning_rate=0.28,
                        # random_state=321)

        clf_gbdt = GradientBoostingClassifier()
                        # class_weight='balanced',
                        # max_depth=5,
                        # criterion='entropy')

        clf_rf = RandomForestClassifier()
                        # n_estimators=15,
                        # class_weight='balanced',
                        # max_depth=5)
        clf_xgb = XGBClassifier()
                        # learning_rate=0.1,
                        # n_estimators=70,
                        # max_depth=4,
                        # min_child_weight=1,
                        # gamma=0,
                        # objective='binary:logistic',
                        # # subsample=0.6,
                        # # colsample_bytree=0.4,
                        # reg_lambda=0.1)

        results = {}
        # for clf in [clf_xgb]:
        for clf in [clf_dt, clf_lr, clf_aba, clf_gbdt, clf_xgb, clf_rf]:
            clf_name = clf.__class__.__name__
            results[clf_name] = {}
            results[clf_name], roc_list = ModelAll.model_all_own(clf, X_under_sample_train, y_under_sample_train,
                                                                 X_test, y_test)
            # 绘制ROC曲线图
            PlotTools.plot_roc_curve2(roc_list['fpr'], roc_list['tpr'], clf_name)

        dt_pd = pd.DataFrame(results['DecisionTreeClassifier'], index=['DT'])
        lr_pd = pd.DataFrame(results['LogisticRegression'], index=['LR'])
        ada_pd = pd.DataFrame(results['AdaBoostClassifier'], index=['ADA'])
        gbdt_pd = pd.DataFrame(results['GradientBoostingClassifier'], index=['GBDT'])
        rf_pd = pd.DataFrame(results['RandomForestClassifier'], index=['RF'])
        xgb_pd = pd.DataFrame(results['XGBClassifier'], index=['XGB'])

        all_pd = pd.concat([lr_pd, dt_pd, ada_pd, gbdt_pd, rf_pd, xgb_pd])
        all_pd.to_csv('data/score3/all_models_os.csv')
        print('结果已保存至score文件夹下 ^_^')
