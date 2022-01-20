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

    df = df.sort_values(by=['stdcal'])

    for group, values in df.groupby(['scaler', 'model', 'splitter']):

        xs = []
        ys = []
        cs = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['nllh'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))

            if (not x) or (not y):
                continue

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
                marker = 'x'
            elif subgroup == 'td':
                marker = '.'
            else:
                marker = '*'

            dens = ax.scatter(
                              x,
                              y,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              )

        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel('Negative Log Likelihood')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'total',
                'nllh',
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'nllh.png')
        fig.savefig(name)

        pl.close('all')

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

        xs = []
        ys = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['nllh'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))

            if (not x) or (not y):
                continue

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
                marker = 'x'
            elif subgroup == 'td':
                marker = '.'
            else:
                marker = '*'

            dens = ax.scatter(
                              x,
                              y,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              )

        ax.legend()
        ax.set_xlabel(xaxis)
        ax.set_ylabel('Negative Log Likelihood')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'groups',
                group[-1],
                'nllh',
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'nllh.png')
        fig.savefig(name)

        pl.close('all')
