from matplotlib import pyplot as pl
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import os


def chunck(x, n):
    '''
    Devide x data into n sized bins.
    '''
    for i in range(0, len(x), n):
        yield x[i:i+n]


def make_plots(save, bin_size):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    xaxis = 'stdcal'

    df = df.sort_values(by=xaxis)

    for group, values in df.groupby(['scaler', 'model', 'splitter']):

        # For the ideal calibration line
        maxx = []
        maxy = []
        minx = []
        miny = []

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['y'].values-subvalues['y_pred'].values
            c = subvalues['pdf'].values

            x = chunck(x, bin_size)
            y = chunck(y, bin_size)
            c = chunck(c, bin_size)

            x = [np.ma.mean(i) for i in x]
            y = [(np.ma.sum(i**2)/len(i))**0.5 for i in y]
            c = [np.ma.prod(i) for i in c]

            if subgroup is True:
                marker = '1'
            else:
                marker = '.'

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='In Domain: {}'.format(subgroup),
                              cmap=pl.get_cmap('viridis'),
                              )

            maxx.append(max(x))
            maxy.append(max(y))
            minx.append(min(x))
            miny.append(min(y))

        maxx = max(maxx)
        maxy = max(maxy)
        minx = min(minx)
        miny = min(miny)

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{c}$')
        ax.set_ylabel('RMS residuals')

        fig.tight_layout()
        fig.colorbar(dens)

        name = '_'.join(group[:3])
        name = [save, 'aggregate', name, 'total', 'calibration']
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

        # For the ideal calibration line
        maxx = []
        maxy = []
        minx = []
        miny = []

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['y'].values-subvalues['y_pred'].values
            c = subvalues['pdf'].values

            x = chunck(x, bin_size)
            y = chunck(y, bin_size)
            c = chunck(c, bin_size)

            x = [np.ma.mean(i) for i in x]
            y = [(np.ma.sum(i**2)/len(i))**0.5 for i in y]
            c = [np.ma.prod(i) for i in c]

            if subgroup is True:
                marker = '1'
            else:
                marker = '.'

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='In Domain: {}'.format(subgroup),
                              cmap=pl.get_cmap('viridis'),
                              )

            maxx.append(max(x))
            maxy.append(max(y))
            minx.append(min(x))
            miny.append(min(y))

        maxx = max(maxx)
        maxy = max(maxy)
        minx = min(minx)
        miny = min(miny)

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{c}$')
        ax.set_ylabel('RMS residuals')

        fig.colorbar(dens)
        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [save, 'aggregate', name, 'groups', group[-1], 'calibration']
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')
