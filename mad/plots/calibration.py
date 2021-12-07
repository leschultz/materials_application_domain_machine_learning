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

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

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
            c = [np.ma.mean(i) for i in c]

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

        ax.legend()
        ax.set_xlabel(r'$\sigma_{c}$')
        ax.set_ylabel('RMS residuals')

        fig.colorbar(dens)

        name = '_'.join(group[:3])
        name = [save, 'aggregate', name, 'groups', group[-1]]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'calibration.png')
        fig.savefig(name)

        pl.close('all')
