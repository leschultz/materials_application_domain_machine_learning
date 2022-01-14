from matplotlib import pyplot as pl
from mad.functions import chunck

import matplotlib.colors as colors
import pandas as pd
import numpy as np
import matplotlib
import seaborn
import os


def make_plots(save):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)
    xaxis = 'nllh'

    for group, values in df.groupby(['scaler', 'model', 'splitter']):

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            subgroup = subgroup.upper()
            mean = np.mean(x)
            label = 'Domain: {}, Mean: {:.2f}'.format(subgroup, mean)
            ax = seaborn.distplot(
                                  a=x,
                                  label=label,
                                  )

        ax.legend()
        ax.set_xlabel('Negative Log Likelihood')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'total',
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, xaxis+'.png')
        fig.savefig(name)

        pl.close('all')

    for group, values in df.groupby(['scaler', 'model', 'splitter', 'domain']):

        fig, ax = pl.subplots()
        for subgroup, subvalues in values.groupby('in_domain'):

            x = subvalues[xaxis].values

            if subgroup == 'id':
                marker = '1'
            elif subgroup == 'ud':
                marker = '.'
            else:
                marker = '+'

            subgroup = subgroup.upper()
            mean = np.mean(x)
            label = 'Domain: {}, Mean: {:.2f}'.format(subgroup, mean)
            ax = seaborn.distplot(
                                  a=x,
                                  label=label,
                                  )

        ax.legend()
        ax.set_xlabel('Negative Log Likelihood')

        fig.tight_layout()

        name = '_'.join(group[:3])
        name = [
                save,
                'aggregate',
                name,
                'groups',
                group[-1],
                xaxis
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, xaxis+'.png')
        fig.savefig(name)

        pl.close('all')
