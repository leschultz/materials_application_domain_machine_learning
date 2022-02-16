from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as pl

import pandas as pd
import numpy as np
import json
import os


def make_plot(save, score):
    '''
    Positive class is out of domain
    '''

    df = pd.read_csv(os.path.join(save, 'aggregate/data.csv'))
    df = df.loc[df['in_domain'] != 'td']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df[score].notna()]

    y_true = [0 if i == 'id' else 1 for i in df['in_domain'].values]
    scaler = StandardScaler()

    if (score == 'pdf') | (score == 'logpdf'):
        sign = -1
    else:
        sign = 1

    data = {}

    for cols in [[score], [score, 'stdcal']]:

        score_name = '_'.join(cols)

        y_scores = df[cols].values
        y_scores[:, 0] = y_scores[:, 0]*sign

        scaler.fit(y_scores)
        y_scores = scaler.transform(y_scores)
        y_scores = np.mean(y_scores, axis=1)

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

        data[score_name]['auc'] = auc_score
        data[score_name]['max_f1'] = max_f1
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
