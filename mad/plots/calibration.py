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
        vmin = []
        vmax = []

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = abs(subvalues['y'].values-subvalues['y_pred'].values)
            c = subvalues['pdf'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            std = np.ma.std(subvalues['y'].values)
            x = np.array([np.ma.mean(i) for i in x])
            y = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in y])
            c = [np.ma.mean(np.ma.log(i)) for i in c]

            # Normalization
            x = x/std
            y = y/std

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='Domain: {}'.format(subgroup),
                              cmap=pl.get_cmap('viridis'),
                              )

            maxx.append(np.ma.max(x))
            maxy.append(np.ma.max(y))
            minx.append(np.ma.min(x))
            miny.append(np.ma.min(y))
            vmin.append(np.ma.min(c))
            vmax.append(np.ma.max(c))

        vmin = np.ma.min(vmin)
        vmax = np.ma.max(vmax)
        maxx = np.ma.max(maxx)
        maxy = np.ma.max(maxy)
        minx = np.ma.min(minx)
        miny = np.ma.min(miny)

        normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        dens.norm = normalize

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{c}/\sigma_{y}$')
        ax.set_ylabel(r'RMSE/$\sigma_{y}$')

        cbar = fig.colorbar(dens)
        cbar.set_label('Log Likelihood')
        fig.tight_layout()

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
        vmin = []
        vmax = []

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = abs(subvalues['y'].values-subvalues['y_pred'].values)
            c = subvalues['pdf'].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            std = np.ma.std(subvalues['y'].values)
            x = [np.ma.mean(i) for i in x]
            y = [(np.ma.sum(i**2)/len(i))**0.5 for i in y]
            c = [np.ma.mean(np.ma.log(i)) for i in c]

            # Normalization
            x = x/std
            y = y/std

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='Domain: {}'.format(subgroup),
                              cmap=pl.get_cmap('viridis'),
                              )

            maxx.append(np.ma.max(x))
            maxy.append(np.ma.max(y))
            minx.append(np.ma.min(x))
            miny.append(np.ma.min(y))
            vmin.append(np.ma.min(c))
            vmax.append(np.ma.max(c))

        vmin = np.ma.min(vmin)
        vmax = np.ma.max(vmax)
        maxx = np.ma.max(maxx)
        maxy = np.ma.max(maxy)
        minx = np.ma.min(minx)
        miny = np.ma.min(miny)

        normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        dens.norm = normalize

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{c}/\sigma_{y}$')
        ax.set_ylabel(r'RMSE/$\sigma_{y}$')

        cbar = fig.colorbar(dens)
        cbar.set_label('Log Likelihood')
        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [save, 'aggregate', name, 'groups', group[-1], 'calibration']
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')
