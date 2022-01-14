from matplotlib import pyplot as pl
from mad.functions import chunck

import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib
import os


def make_plots(save, bin_size, xaxis):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    ares = abs(df['y']-df['y_pred'])  # Absolute residuals
    errs = abs(ares-df['std'])  # Calibration error
    errs = errs/std  # Normalization

    df['ares'] = ares
    df['err_in_err'] = errs
    df = df.sort_values(by=['stdcal', 'ares'])

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

            # Mask values
            x = np.ma.array(x, mask=np.isnan(x))
            y = np.ma.array(y, mask=np.isnan(y))

            x = np.ma.mean(x, axis=1)
            y = np.ma.mean(y, axis=1)

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
        ax.set_ylabel(r'|RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$|')

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

            # Mask values
            x = np.ma.array(x, mask=np.isnan(x))
            y = np.ma.array(y, mask=np.isnan(y))

            x = np.ma.mean(x, axis=1)
            y = np.ma.mean(y, axis=1)

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
        ax.set_ylabel(r'|RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$|')

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
