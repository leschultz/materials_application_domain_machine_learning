from matplotlib import pyplot as pl
from mad.functions import chunck
from sklearn import metrics

import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib
import os


def make_plots(save, bin_size, xaxis, dist):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    df['ares'] = abs(df['y'].values-df['y_pred'].values)
    df = df.sort_values(by=[xaxis, 'ares', dist])

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

        rows = []
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['ares'].values
            c = subvalues[dist].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            if (not x) or (not y) or (not c):
                continue

            # Mask values
            x = np.ma.array(x, mask=np.isnan(x))
            y = np.ma.array(y, mask=np.isnan(y))
            c = np.ma.array(c, mask=np.isnan(c))

            x = np.array([np.ma.mean(i) for i in x])
            y = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in y])

            if dist == 'pdf':
                c = [np.ma.mean(np.ma.log(i)) for i in np.ma.masked_invalid(c)]
                dist_label = 'Log Likelihood'
            else:
                c = [np.ma.mean(i) for i in np.ma.masked_invalid(c)]
                dist_label = dist

            # Normalization
            x = x/std
            y = y/std

            # Table data
            mae = metrics.mean_absolute_error(y, x)
            rmse = metrics.mean_squared_error(y, x)**0.5
            r2 = metrics.r2_score(y, x)

            domain_name = subgroup.upper()
            domain_name = '{}'.format(domain_name)
            rmse = '{:.2E}'.format(rmse)
            mae = '{:.2E}'.format(mae)
            r2 = '{:.2f}'.format(r2)

            rows.append([domain_name, rmse, mae, r2])

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
                zorder = 3
            elif subgroup == 'ud':
                marker = 'x'
                zorder = 2
            elif subgroup == 'td':
                marker = '.'
                zorder = 1
            else:
                marker = '*'
                zorder = 0

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              cmap=pl.get_cmap('viridis'),
                              vmin=vmin,
                              vmax=vmax,
                              zorder=zorder
                              )

        ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

        ax.set_xlim([minx-0.1*abs(minx), maxx+0.1*abs(maxx)])
        ax.set_ylim([miny-0.1*abs(minx), maxy+0.1*abs(maxx)])

        ax.legend()
        ax.set_xlabel(r'$\sigma_{m}/\sigma_{y}$')
        ax.set_ylabel(r'RMSE/$\sigma_{y}$')

        cbar = fig.colorbar(dens)
        cbar.set_label(dist_label)

        # Make a table
        table = ax.table(
                         cellText=rows,
                         colLabels=[r'Domain', r'RMSE', r'MAE', r'$R^2$'],
                         colWidths=[0.15]*3+[0.1],
                         loc='lower right',
                         )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.25, 1.25)

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

        rows = []
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values
            y = subvalues['ares'].values
            c = subvalues[dist].values

            x = list(chunck(x, bin_size))
            y = list(chunck(y, bin_size))
            c = list(chunck(c, bin_size))

            if (not x) or (not y) or (not c):
                continue

            # Mask values
            x = np.ma.array(x, mask=np.isnan(x))
            y = np.ma.array(y, mask=np.isnan(y))
            c = np.ma.array(c, mask=np.isnan(c))

            x = [np.ma.mean(i) for i in x]
            y = [(np.ma.sum(i**2)/len(i))**0.5 for i in y]

            if dist == 'pdf':
                c = [np.ma.mean(np.ma.log(i)) for i in np.ma.masked_invalid(c)]
                dist_label = 'Log Likelihood'
            else:
                c = [np.ma.mean(i) for i in np.ma.masked_invalid(c)]
                dist_label = dist

            # Normalization
            x = x/std
            y = y/std

            # Table data
            mae = metrics.mean_absolute_error(y, x)
            rmse = metrics.mean_squared_error(y, x)**0.5
            r2 = metrics.r2_score(y, x)

            domain_name = subgroup.upper()
            domain_name = '{}'.format(domain_name)
            rmse = '{:.2E}'.format(rmse)
            mae = '{:.2E}'.format(mae)
            r2 = '{:.2f}'.format(r2)

            rows.append([domain_name, rmse, mae, r2])

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
                zorder = 3
            elif subgroup == 'ud':
                marker = 'x'
                zorder = 2
            elif subgroup == 'td':
                marker = '.'
                zorder = 1
            else:
                marker = '*'
                zorder = 0

            dens = ax.scatter(
                              x,
                              y,
                              c=c,
                              marker=marker,
                              label='Domain: {}'.format(subgroup.upper()),
                              cmap=pl.get_cmap('viridis'),
                              vmin=vmin,
                              vmax=vmax,
                              zorder=zorder
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

        # Make a table
        table = ax.table(
                         cellText=rows,
                         colLabels=[r'Domain', r'RMSE', r'MAE', r'$R^2$'],
                         colWidths=[0.15]*3+[0.1],
                         loc='lower right',
                         )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.25, 1.25)

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
