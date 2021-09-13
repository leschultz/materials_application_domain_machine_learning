from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def make_plots(save, show=False):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'spliter', 'features']

    df = pd.read_csv(os.path.join(path, 'data_stats.csv'))
    for group, values in df.groupby(groups):

        vals = sorted(values['logpdf_mean'].values)
        fig, ax = pl.subplots()
        ax.bar(range(len(vals)), vals)

        ax.set_ylabel('Counts')
        ax.set_xlabel('Mean logpdf')
        fig.tight_layout()

        group = list(map(str, group))
        new_path = os.path.join(path, '_'.join(group))
        os.makedirs(new_path, exist_ok=True)

        fig.savefig(os.path.join(
                                 new_path,
                                 'logpdf'
                                 ))

        if show:
            pl.show()
