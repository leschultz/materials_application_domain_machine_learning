from sklearn.metrics import (
                             precision_recall_curve,
                             PrecisionRecallDisplay,
                             average_precision_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             )

from matplotlib import pyplot as pl

from itertools import groupby
from functools import reduce
from sklearn import metrics
from scipy import stats

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib
import warnings
import json
import os

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def parity(
           y,
           y_pred,
           sigma_y,
           d,
           save='.',
           suffix='',
           ):
    '''
    Make a parity plot.

    inputs:
        y = The true target value.
        y_pred = The predicted target value.
        sigma_y = The standard deviation of target variable.
        d = Dissimilarity measure
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sigma = metrics.mean_squared_error(y/sigma_y, y_pred/sigma_y)**0.5

    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)

    label = r'$RMSE/\sigma_{y}=$'
    label += r'{:.2}'.format(rmse_sigma)
    label += '\n'
    label += r'$RMSE=$'
    label += r'{:.2}'.format(rmse)
    label += '\n'
    label += r'$MAE=$'
    label += r'{:.2}'.format(mae)
    label += '\n'
    label += r'$R^{2}=$'
    label += r'{:.2}'.format(r2)

    fig, ax = pl.subplots()

    sc = ax.scatter(
                    y,
                    y_pred,
                    c=d,
                    cmap='viridis',
                    marker='.',
                    zorder=2,
                    label=label,
                    vmin=0.0,
                    vmax=1.0,
                    )

    bar = fig.colorbar(sc, ax=ax, label='D')

    limits = []
    min_range = min(min(y), min(y_pred))
    max_range = max(max(y), max(y_pred))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)

    # Line of best fit
    ax.plot(
            limits,
            limits,
            label=r'$y=\hat{y}$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_ylabel(r'$\hat{y}$')
    ax.set_xlabel('y')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    fig.savefig(os.path.join(
                             save,
                             'parity_{}.png'.format(suffix)
                             ), bbox_inches='tight')
    fig_legend.savefig(os.path.join(
                                    save,
                                    'parity_{}_legend.png'.format(suffix)
                                    ), bbox_inches='tight')

    pl.close(fig)
    pl.close(fig_legend)

    data = {}
    data[r'$RMSE$'] = rmse
    data[r'$RMSE/\sigma_{y}$'] = rmse_sigma
    data[r'$MAE$'] = mae
    data[r'$R^{2}$'] = r2
    data['y'] = y.tolist()
    data['y_pred'] = y_pred.tolist()

    jsonfile = os.path.join(save, 'parity_{}.json'.format(suffix))
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def cdf(x, save=None, binsave=None, subsave='', choice='standard_normal'):
    '''
    Plot the quantile quantile plot for cummulative distributions.

    inputs:
        x = The residuals normalized by the calibrated uncertainties.
        save = The location to save the figure/data.
        binsave = Adding to a directory of the saving for each bin.
        subsave = Append a name to the save file.
        choice = Whether to compare to standard normal distribution,
                 same mean variance of 1 for z, zero mean variance of z for z.

    outputs:
        y = The cummulative distribution of observed data.
        y_pred = The cummulative distribution of standard normal distribution.
        area = The area between y and y_pred.
    '''

    cdf_name = 'cdf'
    parity_name = 'cdf_parity'
    dist_name = 'distribution'
    if binsave is not None:
        save = os.path.join(save, 'each_bin')
        cdf_name = '{}_{}'.format(cdf_name, binsave)
        parity_name = '{}_{}'.format(parity_name, binsave)
        dist_name = '{}_{}'.format(dist_name, binsave)

    area_label = 'Observed Distribution'
    area_label += '\nMiscalibration Area: {:.3f}'.format(areaparity)

    fig, ax = pl.subplots()

    ax.plot(
            y,
            y_pred,
            zorder=0,
            color='b',
            linewidth=4,
            label=area_label,
            )

    # Line of best fit
    ax.plot(
            y,
            y,
            color='k',
            linestyle=':',
            zorder=1,
            linewidth=4,
            label='Ideal',
            )

    ax.set_ylabel('Predicted CDF')
    ax.set_xlabel('Standard Normal CDF')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    ax.set_aspect('equal')
    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    fig.savefig(os.path.join(
                             save,
                             '{}{}.png'.format(parity_name, subsave)
                             ), bbox_inches='tight')

    fig_legend.savefig(os.path.join(
                                    save,
                                    '{}{}_legend.png'.format(
                                                             parity_name,
                                                             subsave
                                                             )
                                    ), bbox_inches='tight')

    pl.close(fig)
    pl.close(fig_legend)

    data = {}
    data['y'] = list(y)
    data['y_pred'] = list(y_pred)
    data['Area'] = areaparity
    with open(os.path.join(
                           save,
                           '{}{}.json'.format(parity_name, subsave)
                           ), 'w') as handle:
        json.dump(data, handle)

    area_label = 'Observed Distribution'
    area_label += '\nMiscalibration Area: {:.3f}'.format(areacdf)

    fig, ax = pl.subplots()

    ax.plot(
            eval_points,
            y,
            zorder=0,
            color='g',
            linewidth=4,
            label='Standard Normal Distribution',
            )

    ax.plot(
            eval_points,
            y_pred,
            zorder=1,
            color='r',
            linewidth=4,
            label=area_label,
            )

    ax.set_xlabel('z')
    ax.set_ylabel('CDF(z)')

    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    fig.savefig(os.path.join(
                             save,
                             '{}{}.png'.format(cdf_name, subsave),
                             ), bbox_inches='tight')

    fig_legend.savefig(os.path.join(
                                    save,
                                    '{}{}_legend.png'.format(
                                                             cdf_name,
                                                             subsave
                                                             ),
                                    ), bbox_inches='tight')

    pl.close(fig)
    pl.close(fig_legend)

    data = {}
    data['x'] = list(eval_points)
    data['y'] = list(y)
    data['y_pred'] = list(y_pred)
    data['Area'] = areacdf
    with open(os.path.join(
                           save,
                           '{}{}.json'.format(cdf_name, subsave),
                           ), 'w') as handle:
        json.dump(data, handle)


def bins(d_min, d_mean, d_max, e, elabel, gt, ylabel, save, suffix):
    '''
    Plot statistical errors with respect to dissimilarity.

    inputs:
        d_min = The minimum of each bin.
        d_mean = The mean of each bin.
        d_max = The max of each bin.
        e = The error statistic.
        elabel = The domain labels.
        gt = The domain ground truth.
        ylabel = The y-axis label.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    data = {'gt': gt}
    fig, ax = pl.subplots()
    for dom in [True, False]:

        if dom is True:
            color = 'g'
            marker = '.'
            label = 'ID'
        else:
            color = 'r'
            marker = 'x'
            label = 'OD'

        # Domain indexes
        dom = dom == elabel

        ax.scatter(
                   d_mean[dom],
                   e[dom],
                   marker=marker,
                   color=color,
                   label='{} Mean'.format(label),
                   )

        ax.scatter(
                   d_min[dom],
                   e[dom],
                   marker=1,
                   color=color,
                   label='{} Start'.format(label),
                   )

        ax.scatter(
                   d_max[dom],
                   e[dom],
                   marker='|',
                   color=color,
                   label='{} End'.format(label),
                   )

        data[marker] = {}
        data[marker][label] = {}
        data[marker][label]['min_d'] = d_min[dom].tolist()
        data[marker][label]['mean_d'] = d_mean[dom].tolist()
        data[marker][label]['max_d'] = d_max[dom].tolist()
        data[marker][label]['e'] = e[dom].tolist()

    ax.axhline(
               gt,
               color='r',
               label='GT = {:.2f}'.format(gt),
               )

    ax.set_ylabel(ylabel)
    ax.set_xlabel('D')

    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    fig.savefig(os.path.join(
                             save,
                             'bins_{}.png'.format(suffix)
                             ), bbox_inches='tight')
    fig_legend.savefig(os.path.join(
                                    save,
                                    'bins_{}_legend.png'.format(suffix)
                                    ), bbox_inches='tight')

    pl.close(fig)
    pl.close(fig_legend)

    jsonfile = os.path.join(save, 'bins_{}.json'.format(suffix))
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def pr(df, save, suffix):
    '''
    Plot PR curve and acquire thresholds.

    inputs:
        df = Data with relevant pr data.
        save = The locatin to save the figure/data.
        suffix = Append a suffix to the save name.

    outputs:
        custom = Data containing threholds for choice of precision/score.
    '''

    precision = df['Precision']
    recall = df['Recall']
    auc_score = df['AUC']
    diff = df['AUC-Baseline']
    baseline = df['Baseline']

    fig, ax = pl.subplots()

    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_label = 'AUC: {:.2f}\n'.format(auc_score)
    pr_label += 'AUC-Baseline: {:.2f}'.format(diff)

    pr_display.plot(ax=ax, label=pr_label)

    ax.hlines(
              baseline,
              color='r',
              linestyle=':',
              label='Baseline: {:.2f}'.format(baseline),
              xmin=0.0,
              xmax=1.0,
              )

    skip = [
            'Precision',
            'Recall',
            'Thresholds',
            'AUC',
            'Baseline',
            'AUC-Baseline',
            ]

    for key, values in df.items():

        if key in skip:
            continue

        p = df[key]['Precision']
        r = df[key]['Recall']
        t = df[key]['Threshold']

        label = key
        label += '\nPrecision: {:.2f}'.format(p)
        label += '\nRecall: {:.2f}'.format(r)
        label += '\nThreshold: {:.2f}'.format(t)
        ax.scatter(r, p, marker='o', label=label)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()

    fig_legend, ax_legend = pl.subplots()
    ax_legend.axis(False)
    legend = ax_legend.legend(
                              *ax.get_legend_handles_labels(),
                              frameon=False,
                              loc='center',
                              bbox_to_anchor=(0.5, 0.5)
                              )
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)

    legend = ax.legend()
    legend.remove()
    fig.savefig(os.path.join(
                             save,
                             'pr_{}.png'.format(suffix)
                             ), bbox_inches='tight')
    fig_legend.savefig(os.path.join(
                                    save,
                                    'pr_{}_legend.png'.format(suffix)
                                    ), bbox_inches='tight'
                       )
    pl.close(fig)
    pl.close(fig_legend)

    # Repare plot data for saving
    jsonfile = os.path.join(save, 'pr_{}.json'.format(suffix))
    with open(jsonfile, 'w') as handle:
        json.dump(df, handle)


def confusion(y_true, y_pred, pos_label, save='.'):

    if pos_label == 'id':
        labels = ['OD', 'ID']
    else:
        labels = ['ID', 'OD']
        y_true = ~y_true

    conf = confusion_matrix(y_true, y_pred)

    # In case only one class exists
    if conf.shape == (1, 1):

        t = list(set(y_true))[0]
        p = list(set(y_pred))[0]

        if (t == p) and (t == 0):
            conf = np.array([[conf[0, 0], 0], [0, 0]])
        elif (t == p) and (t == 1):
            conf = np.array([[0, 0], [0, conf[0, 0]]])
        else:
            raise 'You done fucked up'

    fig, ax = pl.subplots()
    disp = ConfusionMatrixDisplay(conf, display_labels=labels)
    disp.plot(ax=ax)
    fig_data = conf.tolist()

    disp.figure_.savefig(
                         save+'_confusion.png',
                         bbox_inches='tight'
                         )
    pl.close(fig)

    jsonfile = save+'_confusion.json'
    with open(jsonfile, 'w') as handle:
        json.dump(fig_data, handle)


def generate_confusion(df, dfbin, save=None):
    '''
    Generate confusion matrix for predictions compared to ground truth.

    inputs:
        df = Dataframe for single predictions.
        dfbin = Dataframe for statistical predictions.
        save = The location to save figures.
    '''

    def ds(x):
        return [i for i in df.columns if x in i]

    ids = ds('ID')
    ods = ds('OD')
    ds = ids+ods

    for i in ds:

        if 'BIN' in i:
            newsave = os.path.join(save, 'intervals')
        else:
            newsave = os.path.join(save, 'single')

        if 'y_stdc/std(y)' in i:
            newsave = os.path.join(newsave, 'y_stdc_std(y)')
        elif 'dist' in i:
            newsave = os.path.join(newsave, 'dist')

        if 'ID' in i:
            newsave = os.path.join(newsave, 'id')
            pos_label = 'id'
        elif 'OD' in i:
            newsave = os.path.join(newsave, 'od')
            pos_label = 'od'

        if 'Max F1' in i:
            th = 'Max_F1'
        else:
            th = i.split(' ')[-1]

        newsave = os.path.join(newsave, th)

        # Cannot assign ground truth prior to observation for statistical test
        if 'BIN' in i:
            continue
        else:
            confusion(df['id'], df[i], pos_label, newsave)


class plotter:

    def __init__(self, df, df_bin, save):

        self.df = df
        self.df_bin = df_bin
        self.save = save

    def parity(self):

        # ID/OD data via gt_rmse
        df = self.df[self.df['domain_rmse/sigma_y'] == True]
        parity(
               df.y,
               df.y_pred,
               df.std_y,
               df.d_pred,
               self.save,
               'rmse_id',
               )
        
        df = self.df[self.df['domain_rmse/sigma_y'] == False]
        parity(
               df.y, 
               df.y_pred,
               df.std_y,
               df.d_pred,
               self.save,
               'rmse_od',
               )

        # ID/OD data via gt_area
        df = self.df[self.df['domain_cdf_area'] == True]
        parity(
               df.y, 
               df.y_pred, 
               df.std_y, 
               df.d_pred, 
               self.save, 
               'area_id',
               )
        
        df = self.df[self.df['domain_cdf_area'] == False]
        parity(
               df.y,
               df.y_pred,
               df.std_y,
               df.d_pred,
               self.save,
               'area_od',
               )

    def bins(self, gt_rmse, gt_area):

        # Plot data
        d_min = self.df_bin.d_pred_min
        d_mean = self.df_bin.d_pred_mean
        d_max = self.df_bin.d_pred_max
        rmse = self.df_bin['rmse/std_y']
        rmse_label = self.df_bin['domain_rmse/sigma_y']
        area = self.df_bin['cdf_area']
        area_label = self.df_bin['domain_cdf_area']

        # Plotting for rmse
        bins(
             d_min,
             d_mean,
             d_max,
             rmse,
             rmse_label,
             gt_rmse,
             r'$E^{rmse}$',
             self.save,
             'rmse',
             )

        # Plotting for area
        bins(
             d_min,
             d_mean,
             d_max,
             area,
             area_label,
             gt_area,
             r'$E^{area}$',
             self.save,
             'area',
             )

    def pr(self, domain_rmse, domain_area):

        # Plotting for rmse
        pr(domain_rmse, self.save, 'rmse')

        # Plotting for area
        pr(domain_area, self.save, 'area')
