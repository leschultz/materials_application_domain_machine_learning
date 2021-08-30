from matplotlib import pyplot as pl
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import os

from mad.functions import parallel


def make_plots(save):

    path = os.path.join(save, 'aggregate')
    groups = ['scaler', 'model', 'spliter']
    drop_cols = groups+['pipe', 'index']

    df = pd.read_csv(os.path.join(path, 'data.csv'))
    for group, values in df.groupby(groups):

        vals = sorted(values['logpdf'].values)
        fig, ax = pl.subplots()
        ax.bar(range(len(vals)), vals)

        ax.set_ylabel('Counts')
        ax.set_xlabel('logpdf')
        fig.tight_layout()

        fig.savefig(os.path.join(*[
                                   path,
                                   '_'.join(group),
                                   'logpdf'
                                   ]))
