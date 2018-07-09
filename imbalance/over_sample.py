from imblearn.over_sampling import RandomOverSampler
import pandas as pd


class OverSample:

    @staticmethod
    def over_sample_own(X, y):
        num_normal = int(len(y[y.Class == 0]))
        ros = RandomOverSampler(ratio={1: int(num_normal * 1)}, random_state=42)
        X_os_array, y_os_array = ros.fit_sample(X, y.values.ravel())
        X_os = pd.DataFrame(X_os_array,
                            columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                     'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                     'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'normAmount'])
        y_os = pd.DataFrame(y_os_array, columns=['Class'])
        return X_os, y_os
