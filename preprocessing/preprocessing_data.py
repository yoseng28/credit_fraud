from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataPreprocessing:

    @staticmethod
    def amount_standard(data):
        # Amount标准化
        # Amount数值太大，影响特征重要性
        data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
        data = data.drop(['Time', 'Amount'], axis=1)
        # print(data.head(1))
        return data

    @staticmethod
    def read_X_y(data):
        X = data.ix[:, data.columns != 'Class']
        y = data.ix[:, data.columns == 'Class']
        count_class = pd.value_counts(data['Class']).sort_index()
        print('原始数据类的分布：\n%s' % count_class)
        return X, y
