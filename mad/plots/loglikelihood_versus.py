from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def binner(i, data, save, points, sampling):

    os.makedirs(save, exist_ok=True)

    name = os.path.join(save, 'loglikelihood')
    name += '_{}'.format(i)

    df = data[[i, 'loglikelihood_test']].copy()

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

    # Statistics
    llhs = []
    moderrs = []
    bins = []
    counts = []
    for group, values in df.groupby('bin'):

        if values.empty:
            continue

        llh = np.mean(values['loglikelihood_test'].values)
        moderr = np.mean(values[i].values)
        count = values[i].values.shape[0]

        llhs.append(llh)
        moderrs.append(moderr)
        bins.append(group)
        counts.append(count)

    moderrs = np.array(moderrs)
    llhs = np.array(llhs)

    xlabel = '{}'.format(i).capitalize()
    xlabel = xlabel.replace('_', ' ')

    widths = (max(moderrs)-min(moderrs))/len(moderrs)*0.5
    fig, ax = pl.subplots(2)

    ax[0].plot(moderrs, llhs, marker='.', linestyle='none')
    ax[1].bar(moderrs, counts, widths)

    ax[0].set_ylabel(r'$LogLikelihood$')

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Counts')
    ax[1].set_yscale('log')

    fig.tight_layout()
    fig.savefig(name)

    pl.close('all')

    data = {}
    data['Loglikelihood'] = list(llhs)
    data[xlabel] = list(moderrs)
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
        cols = values.columns.tolist()
        cols.remove('y_test')
        cols.remove('y_test_pred')
        cols.remove('std_test_cal')
        cols.remove('loglikelihood_test')
        cols.remove('split_id')

        parallel(
                 binner,
                 cols,
                 data=values,
                 save=os.path.join(path, '_'.join(group)),
                 points=points,
                 sampling=sampling
                 )
