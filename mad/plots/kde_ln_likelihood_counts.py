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

        fig, ax = pl.subplots()
        ax.hist(values['ln_likelihood'], bins=20, density=True)

        ax.set_ylabel('Counts')
        ax.set_xlabel('Sum Ln Likelihood Metric')
        fig.tight_layout()

        fig.savefig(os.path.join(*[
                                   path,
                                   '_'.join(group),
                                   'kde_ln_likelihood'
                                   ]))
