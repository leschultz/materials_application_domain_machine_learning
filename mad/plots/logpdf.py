from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def graphic(save, set_name, show):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'splitter', 'features']

    df = pd.read_csv(os.path.join(path, set_name+'_data_stats.csv'))
    for group, values in df.groupby(groups):

        if 'logpdf_mean' not in values.columns:
            continue

        vals = sorted(values['logpdf_mean'].values)
        fig, ax = pl.subplots()
        ax.bar(range(len(vals)), vals)

        ax.set_ylabel('Counts')
        ax.set_xlabel('Mean logpdf')
        fig.tight_layout()

        group = list(map(str, group))
        group.append(set_name)
        new_path = os.path.join(path, '_'.join(group))
        os.makedirs(new_path, exist_ok=True)

        fig.savefig(os.path.join(
                                 new_path,
                                 'logpdf'
                                 ))

        if show:
            pl.show()


def make_plots(save, show=False):

    graphic(save, 'test', show)
    graphic(save, 'train', show)
