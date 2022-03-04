from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as pl
from mad.functions import chunck
from sklearn import metrics

import matplotlib.colors as colors
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import json
import os


def make_plots(save, bin_size, xaxis, dist, thresh=0.2):

    df = os.path.join(save, 'aggregate/data.csv')
    df = pd.read_csv(df)

    std = np.ma.std(df['y'].values)
    df['ares'] = abs(df['y'].values-df['y_pred'].values)
    df = df.sort_values(by=[xaxis, 'ares', dist])

    if (dist == 'pdf') or (dist == 'logpdf'):
        sign = -1.0
        dist_label = 'negative '+dist
    else:
        sign = 1.0
        dist_label = dist

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
    zs = []

    rows = []
    rows_table = []
    for subgroup, subvalues in df.groupby('in_domain'):

        if subgroup == 'td':
            continue

        x = subvalues[xaxis].values
        y = subvalues['ares'].values
        c = subvalues[dist].values*sign

        x = list(chunck(x, bin_size))
        y = list(chunck(y, bin_size))
        c = list(chunck(c, bin_size))

        # Skip values that are empty
        if (not x) or (not y) or (not c):
            continue

        # Mask values
        x = np.ma.masked_invalid(x)
        y = np.ma.masked_invalid(y)
        c = np.ma.masked_invalid(c)

        x = np.array([np.ma.mean(i) for i in x])
        y = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in y])
        c = np.array([np.ma.mean(i) for i in c])

        # Normalization
        x = x/std
        y = y/std

        z = abs(y-x)

        # Table data
        rmse = metrics.mean_squared_error(y, x)**0.5
        r2 = metrics.r2_score(y, x)

        domain_name = subgroup.upper()
        rows.append([domain_name, rmse, r2])

        domain_name = '{}'.format(domain_name)
        rmse = '{:.2E}'.format(rmse)
        r2 = '{:.2f}'.format(r2)

        rows_table.append([domain_name, rmse, r2])

        maxx.append(np.ma.max(x))
        maxy.append(np.ma.max(y))
        minx.append(np.ma.min(x))
        miny.append(np.ma.min(y))
        vmin.append(np.ma.min(c))
        vmax.append(np.ma.max(c))
        xs.append(x)
        ys.append(y)
        cs.append(c)
        zs.append(z)
        ds.append(subgroup)

    minx = np.append(minx, 0.0)
    miny = np.append(miny, 0.0)

    vmin = np.ma.min(vmin)
    vmax = np.ma.max(vmax)
    maxx = np.ma.max(maxx)
    maxy = np.ma.max(maxy)
    minx = np.ma.min(minx)
    miny = np.ma.min(miny)

    # For plot data export
    data_cal = {}
    data_err = {}

    err_y_label = r'|RMSE/$\sigma_{y}-\sigma_{m}/\sigma_{y}$|'

    fig, ax = pl.subplots()
    fig_err, ax_err = pl.subplots()
    fig_pr, ax_pr = pl.subplots()
    for x, y, c, z, subgroup in zip(xs, ys, cs, zs, ds):

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

        domain = subgroup.upper()
        dens = ax.scatter(
                          x,
                          y,
                          c=c,
                          marker=marker,
                          label='Domain: {}'.format(domain),
                          cmap=pl.get_cmap('viridis'),
                          vmin=vmin,
                          vmax=vmax,
                          zorder=zorder
                          )

        ax_err.scatter(
                       c,
                       z,
                       marker=marker,
                       label='Domain: {}'.format(domain),
                       zorder=zorder
                       )

        data_err[domain] = {}
        data_err[domain][dist_label] = c.tolist()
        data_err[domain][err_y_label] = z.tolist()

        data_cal[domain] = {}
        data_cal[domain][r'$\sigma_{m}/\sigma_{y}$'] = x.tolist()
        data_cal[domain][r'RMSE/$\sigma_{y}$'] = y.tolist()

    ax.axline([0, 0], [1, 1], linestyle=':', label='Ideal', color='k')

    ax.set_xlim([minx-0.1*abs(minx), maxx+0.1*abs(maxx)])
    ax.set_ylim([miny-0.1*abs(minx), maxy+0.1*abs(maxx)])

    ax.legend()
    ax.set_xlabel(r'$\sigma_{m}/\sigma_{y}$')
    ax.set_ylabel(r'RMSE/$\sigma_{y}$')

    ax_err.legend()
    ax_err.set_xlabel(dist_label)
    ax_err.set_ylabel(err_y_label)

    cbar = fig.colorbar(dens)
    cbar.set_label(dist_label)

    # Make a table
    cols = [r'Domain', r'RMSE', r'$R^2$']
    table = ax.table(
                     cellText=rows_table,
                     colLabels=cols,
                     colWidths=[0.15]*2+[0.1],
                     loc='lower right',
                     )

    data_cal['table'] = {}
    data_cal['table']['metrics'] = cols
    data_cal['table']['rows'] = rows

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.25, 1.25)

    fig.tight_layout()
    fig_err.tight_layout()

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'calibration',
            xaxis+'_vs_'+dist
            ]

    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'calibration.png')
    fig_err.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_cal, handle)

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'err_in_err',
            xaxis+'_vs_'+dist
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'err_in_err.png')
    fig_err.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_err, handle)

    # Precision recall for detecting out of domain.
    labels = []
    y_scores = []
    absres = []
    for i, j, k in zip(ds, xs, zs):
        labels += [i]*len(j)
        y_scores += j.tolist()
        absres += k.tolist()

    # Out of domain as postivie (UD).
    y_true = [1 if i >= thresh else 0 for i in absres]

    precision, recall, thresholds = precision_recall_curve(
                                                           y_true,
                                                           y_scores
                                                           )
    data_pr = {}
    data_pr['precision'] = precision.tolist()
    data_pr['recall'] = recall.tolist()
    data_pr['thresholds'] = thresholds.tolist()

    baseline = sum(y_true)/len(y_true)
    auc_score = auc(recall, precision)

    f1_scores = 2*recall*precision/(recall+precision)
    max_f1 = np.nanmax(f1_scores)
    max_f1_threshold = thresholds[np.where(f1_scores == max_f1)][0]

    data_pr['auc'] = auc_score
    data_pr['max_f1'] = max_f1
    data_pr['max_f1_threshold'] = max_f1_threshold
    data_pr['baseline'] = baseline

    ax_pr.plot(
               recall,
               precision,
               color='b',
               label='AUC={:.2f}\nMax F1={:.2f}'.format(auc_score, max_f1)
               )
    ax_pr.axhline(baseline, color='r', linestyle=':', label='Baseline')
    ax_pr.set_xlim(0.0, 1.05)
    ax_pr.set_ylim(0.0, 1.05)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()

    fig_pr.tight_layout()
    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'precision_recall',
            dist
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'precision_recall.png')
    fig_pr.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_pr, handle)

    # Confusion matrix on threshold
    y_pred = [1 if i >= max_f1_threshold else 0 for i in y_scores]
    matrix = confusion_matrix(y_true, y_pred)

    data_conf = {}
    data_conf['matrix'] = matrix.tolist()

    fig, ax = pl.subplots()
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    fig.tight_layout()

    name = [
            save,
            'aggregate',
            'plots',
            'total',
            'confusion',
            dist
            ]
    name = map(str, name)
    name = os.path.join(*name)
    os.makedirs(name, exist_ok=True)
    name = os.path.join(name, 'confusion.png')
    fig.savefig(name)

    # Save plot data
    jsonfile = name.replace('png', 'json')
    with open(jsonfile, 'w') as handle:
        json.dump(data_conf, handle)
