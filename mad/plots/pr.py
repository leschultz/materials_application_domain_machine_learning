from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as pl

import pandas as pd
import numpy as np
import json
import os


def make_plot(save, score, sigma, thresh):
    '''
    Positive class is out of domain
    '''

    df = pd.read_csv(os.path.join(save, 'aggregate/data.csv'))
    df = df.loc[df['in_domain'] != 'td']  # Exclude training
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df[score].notna()]

    absres = abs(df['y']-df['y_pred'])/sigma

    y_true = [1 if i >= thresh else 0 for i in absres]

    if (score == 'pdf') | (score == 'logpdf'):
        sign = -1
    else:
        sign = 1

    data = {}

    for cols in [[score], ['stdcal']]:

        score_name = '_'.join(cols)

        y_scores = df[cols].values
        y_scores[:, 0] = y_scores[:, 0]*sign

        precision, recall, thresholds = precision_recall_curve(
                                                               y_true,
                                                               y_scores
                                                               )

        data[score_name] = {}
        data[score_name]['precision'] = precision.tolist()
        data[score_name]['recall'] = recall.tolist()
        data[score_name]['thresholds'] = thresholds.tolist()

        baseline = sum(y_true)/len(y_true)
        auc_score = auc(recall, precision)

        f1_scores = 2*recall*precision/(recall+precision)
        max_f1 = np.nanmax(f1_scores)
        max_f1_threshold = thresholds[np.where(f1_scores == max_f1)][0]

        data[score_name]['auc'] = auc_score
        data[score_name]['max_f1'] = max_f1
        data[score_name]['max_f1_threshold'] = max_f1_threshold
        data[score_name]['baseline'] = baseline

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
                score_name
                ]
        name = map(str, name)
        name = os.path.join(*name)
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'precision_recall.png')
        fig.savefig(name)

        # Save plot data
        jsonfile = name.replace('png', 'json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)
