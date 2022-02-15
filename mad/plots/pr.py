from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as pl

import pandas as pd
import numpy as np
import os


def make_plot(save, score):
    '''
    Positive class is out of domain
    '''

    df = pd.read_csv(os.path.join(save, 'aggregate/data.csv'))
    df = df[['in_domain', score]]
    df = df.loc[df['in_domain'] != 'td']
    df = df[df[score].notna()]

    y_true = [0 if i == 'id' else 1 for i in df['in_domain'].values]
    y_scores = df[score].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    baseline = sum(y_true)/len(y_true)
    auc_score = auc(recall, precision)

    f1_scores = 2*recall*precision/(recall+precision)
    max_f1 = np.max(f1_scores)

    fig, ax = pl.subplots()

    ax.plot(
            recall,
            precision,
            color='b',
            label='AUC={:.2f}\nMax F1={:.2f}'.format(auc_score, max_f1)
            )
    ax.axhline(baseline, color='r', linestyle=':', label='Baseline')
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()

    fig.tight_layout()
    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'precision_recall',
            score
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'precision_recall.png')
    fig.savefig(name)
