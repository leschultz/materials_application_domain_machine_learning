from matplotlib import pyplot as pl
from scipy.optimize import minimize
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def llh(std, res, x):
    '''
    Compute the log likelihood for a case.
    '''

    total = np.log(2*np.pi)
    total += np.log((x[0]*std+x[1])**2)
    total += (res**2)/((x[0]*std+x[1])**2)

    return total


def set_llh(df, x):
    '''
    Compute the log likelihood for a dataset.
    '''

    std = df['std'].values
    res = df['y_test'].values-df['y_test_pred'].values

    opt = minimize(lambda x: sum(llh(std, res, x)), x, method='nelder-mead')
    a, b = opt.x

    likes = llh(std, res, opt.x)

    return a, b, likes


def binner(i, data, actual, pred, save, points, sampling):

    os.makedirs(save, exist_ok=True)

    name = os.path.join(save, 'calibration')

    df = data[[i, actual, pred]].copy()

    if sampling == 'even':
        df['bin'] = pd.cut(
                           df[i],
                           points,
                           include_lowest=True
                           )

    elif sampling == 'equal':
        df.sort_values(by=i, inplace=True)
        df = np.array_split(df, points)
        count = 0
        for j in df:
            j['bin'] = count
            count += 1

        df = pd.concat(df)

    a, b, likes = set_llh(df, [0, 1])
    df['loglikelihood'] = likes

    # Statistics
    rmses = []
    moderrs = []
    moderrscal = []
    likes = []
    bins = []
    counts = []
    for group, values in df.groupby('bin'):

        if values.empty:
            continue

        x = values[actual].values
        y = values[pred].values

        rmse = metrics.mean_squared_error(x, y)**0.5
        moderr = np.mean(values[i].values)
        moderrcal = a*np.mean(values[i].values)+b
        like = np.mean(values['loglikelihood'].values)
        count = values[i].values.shape[0]

        rmses.append(rmse)
        moderrs.append(moderr)
        moderrscal.append(moderrcal)
        likes.append(like)
        bins.append(group)
        counts.append(count)

    moderrs = np.array(moderrs)
    rmses = np.array(rmses)
    likes = np.array(likes)

    widths = (max(moderrs)-min(moderrs))/len(moderrs)*0.5
    maximum = max([max(moderrs), max(rmses)])

    fig, ax = pl.subplots(3, figsize=(8, 10))

    ax[0].plot(
               moderrs,
               rmses,
               marker='.',
               linestyle='none',
               label='Uncalibrated'
               )
    ax[0].plot(
               moderrscal,
               rmses,
               marker='.',
               linestyle='none',
               label='Calibrated'
               )
    ax[0].plot(
               [0, maximum],
               [0, maximum],
               linestyle=':',
               label='Ideal Calibrated'
               )

    ax[1].plot(likes, rmses, marker='.', linestyle='none')

    ax[2].bar(moderrs, counts, widths)

    ax[0].set_ylabel(r'$RMSE$')
    ax[1].set_ylabel(r'$RMSE$')
    ax[0].legend()

    ax[0].set_xlabel(r'$\sigma_{model}$')
    ax[1].set_xlabel('Mimimized Log Likelihood')

    ax[2].set_ylabel('Counts')
    ax[2].set_yscale('log')

    ax[0].legend()

    fig.tight_layout()
    fig.savefig(name)

    data = {}
    data[r'$RMSE$'] = list(rmses)
    data[r'$\sigma_{model}$_uncalibrated'] = list(moderrs)
    data[r'$\sigma_{model}$_calibrated'] = list(moderrscal)
    data['Counts'] = list(counts)

    jsonfile = name+'.json'
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def make_plots(save, points, sampling):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'spliter']
    drop_cols = groups+['pipe', 'index']

    df = pd.read_csv(os.path.join(path, 'data.csv'))
    for group, values in df.groupby(groups):

        values.drop(drop_cols, axis=1, inplace=True)
        cols = ['std']

        parallel(
                 binner,
                 cols,
                 data=values,
                 actual='y_test',
                 pred='y_test_pred',
                 save=os.path.join(path, '_'.join(group)),
                 points=points,
                 sampling=sampling
                 )
