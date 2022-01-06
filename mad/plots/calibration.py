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


def make_plots(save, bin_size, xaxis, dist):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    df['ares'] = abs(df['y'].values-df['y_pred'].values)
    df = df.sort_values(by=[xaxis, 'ares'])

    for group, values in df.groupby(['scaler', 'model', 'splitter']):

        maxx = []
        maxy = []
        minx = []
        miny = []
        vmin = []
        vmax = []
        xs = []
        ys = []
        cs = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['ares'].values
            c = subvalues[dist].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            x = np.array([np.ma.mean(i) for i in x])
            y = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in y])

            if dist == 'pdf':
                c = [np.ma.mean(np.ma.log(i)) for i in c]
                dist_label = 'Log Likelihood'
            else:
                c = [np.ma.mean(i) for i in c]
                dist_label = dist

            # Normalization
            x = x/std
            y = y/std

            maxx.append(np.ma.max(x))
            maxy.append(np.ma.max(y))
            minx.append(np.ma.min(x))
            miny.append(np.ma.min(y))
            vmin.append(np.ma.min(c))
            vmax.append(np.ma.max(c))
            xs.append(x)
            ys.append(y)
            cs.append(c)
            ds.append(subgroup)

        minx = np.append(minx, 0.0)
        miny = np.append(miny, 0.0)

        vmin = np.ma.min(vmin)
        vmax = np.ma.max(vmax)
        maxx = np.ma.max(maxx)
        maxy = np.ma.max(maxy)
        minx = np.ma.min(minx)
        miny = np.ma.min(miny)

        fig, ax = pl.subplots()
        for x, y, c, subgroup in zip(xs, ys, cs, ds):

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
                              vmin=vmin,
                              vmax=vmax,
                              )

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx-0.1*abs(minx), maxx+0.1*abs(maxx)])
        ax.set_ylim([miny-0.1*abs(minx), maxy+0.1*abs(maxx)])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{m}/\sigma_{y}$')
        ax.set_ylabel(r'RMSE/$\sigma_{y}$')

        cbar = fig.colorbar(dens)
        cbar.set_label(dist_label)
        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'total',
                'calibration',
                xaxis+'_vs_'+dist
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

        maxx = []
        maxy = []
        minx = []
        miny = []
        vmin = []
        vmax = []
        xs = []
        ys = []
        cs = []
        ds = []

        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['ares'].values
            c = subvalues[dist].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            x = [np.ma.mean(i) for i in x]
            y = [(np.ma.sum(i**2)/len(i))**0.5 for i in y]

            if dist == 'pdf':
                c = [np.ma.mean(np.ma.log(i)) for i in c]
                dist_label = 'Log Likelihood'
            else:
                c = [np.ma.mean(i) for i in c]
                dist_label = dist

            # Normalization
            x = x/std
            y = y/std

            maxx.append(np.ma.max(x))
            maxy.append(np.ma.max(y))
            minx.append(np.ma.min(x))
            miny.append(np.ma.min(y))
            vmin.append(np.ma.min(c))
            vmax.append(np.ma.max(c))
            xs.append(x)
            ys.append(y)
            cs.append(c)
            ds.append(subgroup)

        minx = np.append(minx, 0.0)
        miny = np.append(miny, 0.0)

        vmin = np.ma.min(vmin)
        vmax = np.ma.max(vmax)
        maxx = np.ma.max(maxx)
        maxy = np.ma.max(maxy)
        minx = np.ma.min(minx)
        miny = np.ma.min(miny)

        fig, ax = pl.subplots()
        for x, y, c, subgroup in zip(xs, ys, cs, ds):

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
                              vmin=vmin,
                              vmax=vmax,
                              )

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx-0.1*abs(minx), maxx+0.1*abs(maxx)])
        ax.set_ylim([miny-0.1*abs(minx), maxy+0.1*abs(maxx)])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{m}/\sigma_{y}$')
        ax.set_ylabel(r'RMSE/$\sigma_{y}$')

        normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(dens, norm=normalize)
        cbar.set_label(dist_label)
        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'groups',
                group[-1],
                'calibration',
                xaxis+'_vs_'+dist
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')
