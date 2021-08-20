from sklearn import metrics
from pathlib import Path

import pandas as pd
import numpy as np
import os

from mad.functions import parallel


def eval_reg_metrics(groups):
    '''
    Evaluate standard regression prediction metrics.
    '''

    group, df = groups

    y = df['y_test']
    y_pred = df['y_test_pred']

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sig = rmse/np.std(y)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    model, scaler, spliter, split_id = group

    results = {}
    results['model'] = model
    results['scaler'] = scaler
    results['spliter'] = spliter
    results['split_id'] = split_id
    results[r'$RMSE$'] = rmse
    results[r'$RMSE/\sigma$'] = rmse_sig
    results[r'$MAE$'] = mae
    results[r'$R^{2}$'] = r2

    return results


def group_metrics(df, cols):
    '''
    '''

    groups = df.groupby(cols)
    mets = parallel(eval_reg_metrics, groups)
    mets = pd.DataFrame(mets)

    return mets


def stats(df, cols):
    '''
    Get the statistic of a dataframe.
    '''

    groups = df.groupby(cols)
    mean = groups.mean().add_suffix('_mean')
    sem = groups.sem().add_suffix('_sem')
    count = groups.count().add_suffix('_count')
    df = mean.merge(sem, on=cols)
    df = df.merge(count, on=cols)
    df = df.reset_index()

    return df


def folds(save):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
    '''

    path = os.path.join(save, 'splits')
    paths = list(Path(save).rglob('split_*.csv'))

    # Load
    df = parallel(pd.read_csv, paths)
    df = pd.concat(df)

    mets = group_metrics(df, ['model', 'scaler', 'spliter', 'split_id'])

    # Get statistics
    dfstats = stats(df, ['index', 'model', 'scaler', 'spliter'])
    metsstats = mets.drop('split_id', axis=1)
    metsstats = stats(metsstats, ['model', 'scaler', 'spliter'])

    # Save data
    save = os.path.join(save, 'aggregate')
    os.makedirs(save, exist_ok=True)
    df.to_csv(
              os.path.join(save, 'data.csv'),
              index=False
              )
    dfstats.to_csv(
                   os.path.join(save, 'data_stats.csv'),
                   index=False
                   )
    mets.to_csv(
                os.path.join(save, 'metrics.csv'),
                index=False
                )
    metsstats.to_csv(
                     os.path.join(save, 'metrics_stats.csv'),
                     index=False
                     )