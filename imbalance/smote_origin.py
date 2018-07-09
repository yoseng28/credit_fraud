from imblearn.over_sampling import SMOTE

import pandas as pd


class SmoteOrigin:

    @staticmethod
    def smote_own(X_train, y_train):
        sm = SMOTE(ratio={1: 420},
                   # kind='',
                   random_state=0)
        X_smote_train_array, y_smote_train_array = sm.fit_sample(X_train, y_train.values.ravel())
        X_smote_train = pd.DataFrame(X_smote_train_array,
                                     columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                              'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                              'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'normAmount'])
        y_smote_train = pd.DataFrame(y_smote_train_array, columns=['Class'])
        return X_smote_train, y_smote_train
