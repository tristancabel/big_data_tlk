import numpy as np
import pylab as pl


def gb_diagnose(X, gb, params):
    """
    gradiant boosting
    gb: gradiant boosting
    params: paraameters dictionnary of gb
    """
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(gb.staged_decision_function(X_test)):
        test_score[i] = gb.loss_(y_test, y_pred)
    pl.figure(figsize=(12,9))
    pl.title('Deviance')
    pl.plot(np.arange(params['n_estimators']) +1, gb.train_score_, 'b-', label='Training set Deviance')
    pl.plot(np.arange(params['n_estimators']) +1, test_score, 'r-', label='Test set Deviance')
    pl.legend(loc='upper right')
    pl.xlabel('Boosting iterations')
    pl.ylabel('Deviance')
    pl.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler

def gd_fit(df):
    """
    gradiant descent
    """
    #first normalize input data
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
