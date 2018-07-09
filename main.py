import pandas as pd
from preprocessing import preprocessing_data
from result import result_LR, result_XGB, result_all
from tools import tools_plot, tools_data


class Main:
    def __init__(self):
        self.pd_data = pd.read_csv('data/creditcard.csv')
        self.yue_plt = tools_plot.PlotTools
        self.yue_data = tools_data.DataTools
        self.yue_proces = preprocessing_data.DataPreprocessing
        self.yue_result_LR = result_LR.ResultLR
        self.yue_XGB = result_XGB.ResultXGB
        self.yue_all = result_all.ResultAll

    def plt(self):
        # class直方图
        # self.yue_plt.fraud_class_histogram(self.pd_data)
        # self.yue_plt.plot_roc_curve()
        self.yue_plt.plot_xgb_roc_curve('data/score2/new.xlsx')

    def fit_lr(self):
        _ = self.yue_proces.amount_standard(self.pd_data)
        # self.yue_result_LR.LR_origin_data(_)
        # self.yue_result_LR.LR_under_sample(_)
        # self.yue_result_LR.LR_under_sample_test(_)
        self.yue_result_LR.LR_EE_smote(_)

    def fit_xgb(self):
        _ = self.yue_proces.amount_standard(self.pd_data)
        # self.yue_XGB.XGB_over_sample(_)
        self.yue_XGB.XGB_under_sample(_)
        # self.yue_XGB.XGB_smote(_)
        # self.yue_XGB.XGB_EE_smote(_)
        # self.yue_XGB.XGB_EE(_)
        # self.yue_XGB.XGB_origin(_)
        # self.yue_XGB.XGB_under_sample_smote(_)

    def fit_all(self):
        _ = self.yue_proces.amount_standard(self.pd_data)
        self.yue_all.result_all_model(_)


if __name__ == '__main__':
    main = Main()
    main.plt()
    # main.fit_xgb()
    # main.fit_all()
    # main.fit_lr()
