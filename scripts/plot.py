from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import os


def binner(data, cols, actual, pred):

    std = np.std(data[actual].values)
    for i in cols:

        df = data[[i, actual, pred]].copy()
        df['bin'] = pd.cut(
                           df[i],
                           10,
                           include_lowest=True
                           )

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

        xlabel = 'Average {}'.format(i)
        if 'std' in i:
            moderrs /= std
            xlabel += r'$/\sigma$'

        fig, ax = pl.subplots(2)

        ax[0].plot(moderrs, rmses/std, marker='.', linestyle='none')
        ax[1].bar(moderrs, counts, (max(moderrs)-min(moderrs))/len(moderrs)*0.75)

        ax[0].set_ylabel(r'$RMSE/\sigma$')

        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel('Counts')
        ax[1].set_yscale('log')

        fig.tight_layout()
        pl.show()


def main():
    df = '../analysis/data.csv'

    df = pd.read_csv(df)

    cols = list(df.columns)
    cols.remove('index')
    cols.remove('actual')
    cols.remove('gpr_pred')
    cols.remove('rf_pred')

    cols = ['rf_std']
    binner(df, cols, 'actual', 'rf_pred')
    cols = ['gpr_std']
    binner(df, cols, 'actual', 'gpr_pred')


if __name__ == '__main__':
    main()
