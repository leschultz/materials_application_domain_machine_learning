from mad.utils import parallel
from sklearn import metrics

import pandas as pd
import numpy as np


def eval_reg_metrics(groups, cols):
    '''
    Evaluate standard regression prediction metrics.
    '''

    group, df = groups

    y = df['y']
    y_pred = df['y_pred']

    rmse = metrics.mean_squared_error(y, y_pred)**0.5

    if y.shape[0] > 1:
        rmse_sig = rmse/np.std(y)
    else:
        rmse_sig = np.nan

    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    results = {}
    results[r'$RMSE$'] = rmse
    results[r'$RMSE/\sigma$'] = rmse_sig
    results[r'$MAE$'] = mae
    results[r'$R^{2}$'] = r2

    for i, j in zip(cols, group):
        results[i] = j

    return results


def group_metrics(df, cols):
    '''
    Get the metrics statistics.
    '''

    groups = df.groupby(cols, dropna=False)
    mets = parallel(eval_reg_metrics, groups, cols=cols)
    mets = pd.DataFrame(mets)

    return mets


def stats(df, cols, drop=None):
    '''
    Get the statistic of a dataframe.
    '''

    if drop:
        df = df.drop(drop, axis=1)

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)
    df = df.reset_index()

    return df
