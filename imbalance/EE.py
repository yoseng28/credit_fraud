import numpy as np

from tools.tools_data import DataTools


class EE:

    @staticmethod
    def ee_own(X, y):
        number_records_fraud = len(y[y.Class == 1])
        fraud_indices = np.array(y[y.Class == 1].index)
        normal_indices = y[y.Class == 0].index
        # 对负类进行下采样
        random_normal_indices = np.random.choice(normal_indices, int(number_records_fraud*1.05), replace=False)
        random_normal_indices = np.array(random_normal_indices)
        # 负类的index + 下采样后正类的index
        under_ee_indices = np.concatenate([fraud_indices, random_normal_indices])
        # iloc是将序列当作数组来访问，下标会从0开始
        X_ee_sample = X.loc[under_ee_indices, :]
        y_ee_sample = y.loc[under_ee_indices, :]
        print('EE采样后的训练集：')
        DataTools.print_data_ratio(y_ee_sample)
        return X_ee_sample, y_ee_sample
