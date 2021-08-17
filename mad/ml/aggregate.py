from sklearn import metrics

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

    results = {
               'result': [rmse, rmse_sig, mae, r2],
               'metric': [r'$RMSE$', r'$RMSE/\sigma$', r'$MAE$', r'$R^{2}$']
               }

    return results


def group_metrics(df, cols):
    '''
    '''

    groups = df.groupby(cols)
    mets = parallel(eval_reg_metrics, groups)

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

    return df

def outer(save):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        save = The directory to save and where split data are.
    '''

    path = os.path.join(save, 'splits')
    paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))

    df = parallel(pd.read_csv, paths)
    df = pd.concat(df)

    met = group_metrics(df, ['model', 'scaler', 'split'])
    print(met)

    # Get statistics
    dfstats = stats(df, ['index', 'model', 'scaler', 'split'])

    # Save data
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


def folds(save):
    '''
    Define the machine learning workflow with nested cross validation
    for gaussian process regression and random forest.
    '''

    # Nested CV
    outer(save)
