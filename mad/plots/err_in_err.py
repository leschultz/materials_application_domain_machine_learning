from matplotlib import pyplot as pl
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib
import os


def chunck(x, n):
    '''
    Devide x data into n sized bins.
    '''
    for i in range(0, len(x), n):
        x_new = x[i:i+n]

        if len(x_new) == n:
            yield x_new


def make_plots(save, bin_size, xaxis):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    df['err_in_err'] = abs(df['y']-df['y_pred'])/df['stdcal']
    df = df.sort_values(by=[xaxis, 'err_in_err'])

    for group, values in df.groupby(['scaler', 'model', 'splitter']):

        xs = []
        ys = []
        cs = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['err_in_err'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))

            x = np.array([np.ma.mean(i) for i in x])
            y = [np.ma.mean(i) for i in y]

            xs.append(x)
            ys.append(y)
            ds.append(subgroup)

        fig, ax = pl.subplots()
        for x, y, subgroup in zip(xs, ys, ds):

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            dens = ax.scatter(
                              x,
                              y,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              )

        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r'RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'total',
                'err_in_err',
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'err_in_err.png')
        fig.savefig(name)

        pl.close('all')

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

        xs = []
        ys = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['err_in_err'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))

            x = [np.ma.mean(i) for i in x]
            y = [np.ma.mean(i) for i in y]

            xs.append(x)
            ys.append(y)
            ds.append(subgroup)

        fig, ax = pl.subplots()
        for x, y, subgroup in zip(xs, ys, ds):

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            dens = ax.scatter(
                              x,
                              y,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              )

        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel(r'RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'groups',
                group[-1],
                'err_in_err',
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'err_in_err.png')
        fig.savefig(name)

        pl.close('all')
