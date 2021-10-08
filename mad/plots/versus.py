from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def find_bin(df):

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

    else:
        df['bin'] = list(range(df.shape[0]))  # Each point a bin


def binner(i, data, actual, pred, save, points, sampling):

    os.makedirs(save, exist_ok=True)

    name = os.path.join(save, 'residual')
    name += '_{}'.format(i)

    df = data[[i, actual, pred]].copy()
    df = find_bin(df)  # Bin the data

    # Statistics
    rmses = []
    moderrs = []
    bins = []
    counts = []
    for group, values in df.groupby('bin'):

        if values.empty:
            continue

        x = values[actual].values
        y = values[pred].values

        rmse = np.mean(x-y)
        moderr = np.mean(values[i].values)
        count = values[i].values.shape[0]

        rmses.append(rmse)
        moderrs.append(moderr)
        bins.append(group)
        counts.append(count)

    moderrs = np.array(moderrs)
    rmses = np.array(rmses)

    xlabel = '{}'.format(i)
    if 'logpdf' == i:
        xlabel = 'Negative '+xlabel
        moderrs = -1*moderrs
    else:
        xlabel = xlabel.capitalize()
        xlabel = xlabel.replace('_', ' ')

    widths = (max(moderrs)-min(moderrs))/len(moderrs)*0.5
    fig, ax = pl.subplots(2)

    ax[0].plot(moderrs, rmses, marker='.', linestyle='none')
    ax[1].bar(moderrs, counts, widths)

    ax[0].set_ylabel(r'$y-\hat{y}$')

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Counts')
    ax[1].set_yscale('log')

    fig.tight_layout()
    fig.savefig(name)

    pl.close('all')

    data = {}
    data['residual'] = list(rmses)
    data[xlabel] = list(moderrs)
    data['Counts'] = list(counts)

    jsonfile = name+'.json'
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def graphics(save, points, sampling):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'splitter']
    drop_cols = groups+['pipe', 'index']

    test = pd.read_csv(os.path.join(path, 'test_data.csv'))
    train = pd.read_csv(os.path.join(path, 'train_data.csv'))

    test['set'] = 'test'
    train['set'] = 'train'

    df = pd.concat([train, test])

    remove = {'y', 'y_pred', 'split_id', 'flag'}
    for group, values in df.groupby(groups):

        values.drop(drop_cols, axis=1, inplace=True)
        cols = set(values.columns.tolist())
        cols = cols.difference(remove)

        group = list(map(str, group))
        parallel(
                 binner,
                 cols,
                 data=values,
                 actual='y',
                 pred='y_pred',
                 save=os.path.join(path, '_'.join(group)),
                 points=points,
                 sampling=sampling
                 )


def make_plots(save, points=None, sampling=None):
    graphics(save, points, sampling)
