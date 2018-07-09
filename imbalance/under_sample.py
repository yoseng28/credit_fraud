import numpy as np


class UnderSample:

    @staticmethod
    def under_sample_own(X, y):
        # 控制抽样个数
        seed_p = 1
        seed_n = 1
        # 随机抽取正类
        number_records_fraud = int(len(y[y.Class == 1]))
        fraud_indices = np.array(y[y.Class == 1].index)
        random_fraud_indices = np.random.choice(fraud_indices, int(number_records_fraud*seed_p), replace=False)
        # 随机抽取负类
        number_normal = int(len(y[y.Class == 0]))
        normal_indices = y[y.Class == 0].index
        # random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
        random_normal_indices = np.random.choice(normal_indices, int(number_records_fraud*seed_n), replace=False)
        random_normal_indices = np.array(random_normal_indices)

        # 负类的index + 下采样后正类的index
        under_sample_indices = np.concatenate([random_fraud_indices, random_normal_indices])

        # iloc是将序列当作数组来访问，下标会从0开始
        X_under_sample = X.loc[under_sample_indices, :]
        y_under_sample = y.loc[under_sample_indices, :]

        normal_percent = len(y_under_sample[y_under_sample.Class == 0]) / len(y_under_sample)
        print("\nnormal百分比: ", normal_percent)
        fraud_percent = len(y_under_sample[y_under_sample.Class == 1]) / len(y_under_sample)
        print("fraud百分比: ", fraud_percent)
        print("下采样后的数据个数: ", len(y_under_sample))
        print('\n')

        return X_under_sample, y_under_sample
