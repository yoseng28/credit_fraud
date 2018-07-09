import math

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from tools.tools_data import DataTools


class SmoteEE:

    @staticmethod
    def smoteEE_own(X, y):
        number_records_fraud = len(y[y.Class == 1])
        fraud_indices = np.array(y[y.Class == 1].index)
        normal_indices = y[y.Class == 0].index
        # 对负类进行下采样
        random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
        random_normal_indices = np.array(random_normal_indices)
        # 负类的index + 下采样后正类的index
        under_ee_indices = np.concatenate([fraud_indices, random_normal_indices])
        # iloc是将序列当作数组来访问，下标会从0开始
        X_ee_sample = X.loc[under_ee_indices, :]
        y_ee_sample = y.loc[under_ee_indices, :]
        print('EE下采样后的训练集：')
        DataTools.print_data_ratio(y_ee_sample)
        sm = SMOTE(ratio={1: math.ceil(number_records_fraud * 1.5)},
                   # random_state=0,
                   kind='regular',
                   )
        X_ee_smote_train_array, y_ee_smote_train_array = sm.fit_sample(X_ee_sample, y_ee_sample.values.ravel())
        X_ee_smote_train = pd.DataFrame(X_ee_smote_train_array,
                                        columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                                 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                                 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'normAmount'])
        y_ee_smote_train = pd.DataFrame(y_ee_smote_train_array, columns=['Class'])
        return X_ee_smote_train, y_ee_smote_train


