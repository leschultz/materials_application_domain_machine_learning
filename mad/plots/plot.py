from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from ad.functions import parallel


def binner(i, data, actual, pred, std, save, bins=None, samples=None):

    os.makedirs(save, exist_ok=True)

    name = os.path.join(save, pred)
    name += '_{}'.format(i)

    df = data[[i, actual, pred]].copy()

    if bins:
        df['bin'] = pd.cut(
                           df[i],
                           bins,
                           include_lowest=True
                           )
    elif samples:
        df.sort_values(by=i, inplace=True)
        df = np.array_split(df, samples)
        count = 0
        for j in df:
            j['bin'] = count
            count += 1

        df = pd.concat(df)

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

        rmse = metrics.mean_squared_error(x, y)**0.5
        moderr = np.mean(values[i].values)
        count = values[i].values.shape[0]

        rmses.append(rmse)
        moderrs.append(moderr)
        bins.append(group)
        counts.append(count)

    moderrs = np.array(moderrs)
    rmses = np.array(rmses)

    if 'std' in i:
        moderrs /= std
        xlabel = r'Average $\dfrac{\sigma_{model}}{\sigma_{true}}$'
    else:
        xlabel = 'Average {}'.format(i)
        xlabel = xlabel.replace('_', ' ')

    widths = (max(moderrs)-min(moderrs))/len(moderrs)*0.5
    fig, ax = pl.subplots(2)

    rmses_std = rmses/std

    ax[0].plot(moderrs, rmses_std, marker='.', linestyle='none')
    ax[1].bar(moderrs, counts, widths)

    ax[0].set_ylabel(r'$RMSE/\sigma$')

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Counts')
    ax[1].set_yscale('log')

    fig.tight_layout()
    fig.savefig(name)

    data = {}
    data[r'$RMSE/\sigma$'] = list(rmses_std)
    data[xlabel] = list(moderrs)
    data['Counts'] = list(counts)

    jsonfile = name+'.json'
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def main():
    df = '../analysis/data.csv'
    save = '../analysis'
    groups = ['scaler', 'model', 'split']
    drop_cols = groups+['pipe', 'index']

    df = pd.read_csv(df)
    for group, values in df.groupby(groups):

        values.drop(drop_cols, axis=1, inplace=True)
        cols = values.columns.tolist()
        cols.remove('y_test')
        cols.remove('y_test_pred')

        parallel(
                 binner,
                 cols,
                 data=values,
                 actual='y_test',
                 pred='y_test_pred',
                 std=np.std(values['y_test']),
                 save=os.path.join(save, '_'.join(group)),
                 bins=15
                 )


if __name__ == '__main__':
    main()
