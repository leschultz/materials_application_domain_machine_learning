from sklearn.metrics import (
                             PrecisionRecallDisplay,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             )

from matplotlib import pyplot as pl
from madml.models import bin_data
from madml import calculators
from sklearn import metrics

import numpy as np

import matplotlib
import json
import os

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def plot_dump(data, fig, ax, name, save, suffix, legend=True):
    '''
    Function to dump figures.

    inputs:
        data = Data to dump in json file.
        fig = Figure object.
        ax = Axes object.
        name = The name for the plot.
        save = The location to save plot.
        suffix = Append a suffix to the save name.
    '''

    fig.tight_layout()

    if legend:

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

        fig_legend.savefig(os.path.join(
                                        save,
                                        '{}_{}_legend.png'.format(name, suffix)
                                        ), bbox_inches='tight')

        ax.legend([]).set_visible(False)

        pl.close(fig_legend)

    fig.savefig(os.path.join(
                             save,
                             '{}_{}.png'.format(name, suffix)
                             ), bbox_inches='tight')

    pl.close(fig)

    jsonfile = os.path.join(save, '{}_{}.json'.format(name, suffix))
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def parity(
           y,
           y_pred,
           y_stdc_pred,
           r_std_y,
           d,
           save='.',
           suffix='',
           ):
    '''
    Make a parity plot.

    inputs:
        y = The true target value.
        y_pred = The predicted target value.
        y_stdc_pred = The uncertainties in predicted values.
        r_std_y = The residuals normalized by standard deviation.
        d = Dissimilarity measure
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    rmse = metrics.mean_squared_error(y, y_pred)**0.5
    rmse_sigma = (sum(r_std_y**2)/r_std_y.shape[0])**0.5

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

    ax.errorbar(
                y,
                y_pred,
                yerr=y_stdc_pred,
                marker='.',
                fmt='none',
                ecolor='r',
                zorder=-1,
                )

    bar = fig.colorbar(sc, ax=ax, label='D')

    # Line of best fit
    limits = []
    min_range = min(min(y), min(y_pred))
    max_range = max(max(y), max(y_pred))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)
    ax.plot(
            limits,
            limits,
            label=r'$y=\hat{y}$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')

    ax.set_ylabel(r'$\hat{y}$')
    ax.set_xlabel('y')

    data = {}
    data[r'$RMSE$'] = rmse
    data[r'$RMSE/\sigma_{y}$'] = rmse_sigma
    data[r'$MAE$'] = mae
    data[r'$R^{2}$'] = r2
    data['y'] = y.tolist()
    data['y_pred'] = y_pred.tolist()

    plot_dump(data, fig, ax, 'parity', save, suffix)


def cdf(df, gt, save, suffix):
    '''
    Plot the quantile quantile plot for cummulative distributions.

    inputs:
        x = The residuals normalized by the calibrated uncertainties.
        gt = The ground truth to examine.
        save = The location to save the figure/data.
    '''

    data = {}
    fig, ax = pl.subplots()
    for group, values in df.groupby(gt):

        eval_points, y, y_pred, _, areacdf = calculators.cdf(values['z'])

        area_label = '{}: '.format(group)
        area_label += '$E^{{{}}}={:.3f}$'.format(suffix, areacdf)

        color = 'g' if group == 'ID' else 'r'

        ax.plot(
                eval_points,
                y_pred,
                zorder=1,
                color=color,
                linewidth=4,
                label=area_label,
                )

        data[group] = {}
        data[group]['x'] = list(eval_points)
        data[group]['y'] = list(y_pred)
        data[group] = areacdf

    # Just need one set of N(0,1) values.
    ax.plot(
            eval_points,
            y,
            zorder=0,
            color='b',
            linewidth=4,
            label='N(0,1)',
            )

    data['N(0,1)'] = {}
    data['N(0,1)']['x'] = list(eval_points)
    data['N(0,1)']['y'] = list(y)
    data['N(0,1)'] = areacdf

    ax.set_xlabel('z')
    ax.set_ylabel('CDF(z)')

    plot_dump(data, fig, ax, 'cdf', save, suffix)


def bins(df, d, e, elabel, gt, ylabel, save, suffix):
    '''
    Plot statistical errors with respect to dissimilarity.

    inputs:
        d = The dissimilarity.
        e = The error statistic.
        elabel = The domain labels.
        gt = The domain ground truth.
        ylabel = The y-axis label.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    data = {'gt': gt}
    fig, ax = pl.subplots()
    for group, values in df.groupby([elabel, 'bin']):

        dom, _ = group

        if dom == 'ID':
            color = 'g'
            bot = 0
        else:
            color = 'r'
            bot = gt

        x = np.array(values[d], dtype=float)
        y = np.array(values[e], dtype=float)

        ax.plot(x, y, color=color)
        ax.fill_between(
                        x,
                        y,
                        bot,
                        color=color,
                        alpha=0.5,
                        )

        data[dom] = {}
        data[dom]['x'] = x.tolist()
        data[dom]['y'] = y.tolist()

    ax.axhline(
               gt,
               color='g',
               label='GT = {:.2f}'.format(gt),
               )

    ax.set_ylabel(ylabel)
    ax.set_xlabel('D')

    plot_dump(data, fig, ax, 'bins', save, suffix)


def pr(df, save, suffix):
    '''
    Plot PR curve and acquire thresholds.

    inputs:
        df = Data with relevant pr data.
        save = The locatin to save the figure/data.
        suffix = Append a suffix to the save name.
    '''

    precision = df['Precision']
    recall = df['Recall']
    auc_score = df['AUC']
    diff = df['AUC-Baseline']
    baseline = df['Baseline']

    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    label = 'AUC: {:.2f}\n'.format(auc_score)
    label += 'AUC-Baseline: {:.2f}'.format(diff)

    pr_display.plot(label=label)

    fig, ax = pl.gcf(), pl.gca()

    ax.hlines(
              baseline,
              color='r',
              linestyle=':',
              label='Baseline: {:.2f}'.format(baseline),
              xmin=0.0,
              xmax=1.0,
              )

    # Keys that are not prediction thresholds
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

    plot_dump(df, fig, ax, 'pr', save, suffix)


def confusion(y, y_pred, save, suffix):
    '''
    Make a confusion matrix.

    inputs:
        y = The true value.
        y_pred = The predicted value.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    conf = confusion_matrix(y, y_pred, labels=['ID', 'OD'])

    fig, ax = pl.subplots()
    disp = ConfusionMatrixDisplay(conf, display_labels=['ID', 'OD'])
    disp.plot(ax=ax)

    plot_dump(conf.tolist(), fig, ax, 'confusion', save, suffix, legend=False)


def area_vs_rmse(df, save):
    '''
    Make a plot of miscalibration area vs rmse.

    inputs:
        df = Dataframe.
        save = The directory to save plot.
    '''

    d = df['d_pred_mean']
    rmse = df['rmse/std_y']
    area = df['cdf_area']

    fig, ax = pl.subplots()
    sc = ax.scatter(
                    rmse,
                    area,
                    c=d,
                    cmap='viridis',
                    marker='.',
                    zorder=2,
                    label='Bins',
                    vmin=0.0,
                    vmax=1.0,
                    )

    bar = fig.colorbar(sc, ax=ax, label='D')

    ax.set_xlabel(r'$E^{rmse}$')
    ax.set_ylabel(r'$E^{area}$')

    data = {}
    data['x'] = rmse.tolist()
    data['y'] = area.tolist()
    data['d'] = d.tolist()

    plot_dump(data, fig, ax, 'area_vs_rmse', save, '')


def rmse_vs_stdc(df, save, suffix):
    '''
    Make a plot of rmse vs stdc.

    inputs:
        df = Dataframe.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    d = df['d_pred_mean']
    rmse = df['rmse/std_y']
    stdc = df['y_stdc_pred/std_y_mean']

    fig, ax = pl.subplots()
    sc = ax.scatter(
                    stdc,
                    rmse,
                    c=d,
                    cmap='viridis',
                    marker='.',
                    zorder=2,
                    label='Bins',
                    vmin=0.0,
                    vmax=1.0,
                    )

    bar = fig.colorbar(sc, ax=ax, label='D')

    # Line of best fit
    limits = []
    min_range = min(min(rmse), min(stdc))
    max_range = max(max(rmse), max(stdc))
    span = max_range-min_range
    limits.append(min_range-0.1*span)
    limits.append(max_range+0.1*span)
    ax.plot(
            limits,
            limits,
            label=r'$E^{rmse}=mean(\sigma_{c}/\sigma_(y))$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')

    ax.set_xlabel(r'$\sigma_{c}/\sigma(y)$')
    ax.set_ylabel(r'$E^{rmse}$')

    data = {}
    data['x'] = stdc.tolist()
    data['y'] = rmse.tolist()
    data['d'] = d.tolist()

    plot_dump(data, fig, ax, 'rmse_vs_stdc', save, suffix)


class plotter:

    def __init__(
                 self,
                 df,
                 df_bin,
                 domain_rmse,
                 domain_area,
                 gt_rmse,
                 gt_area,
                 save
                 ):

        self.save = save
        self.domains = ['domain_rmse/sigma_y', 'domain_cdf_area']
        self.errors = ['rmse/std_y', 'cdf_area']
        self.assessments = ['rmse', 'area']

        # Domain
        self.gts = [gt_rmse, gt_area]  # Ground truths
        self.prs = [domain_rmse, domain_area]  # Domain pr

        # For plotting purposes
        cols = self.errors+self.domains
        self.df = df.sort_values(by=['d_pred']+cols)
        self.df_bin = df_bin.sort_values(by=['d_pred_max']+cols)

        self.bins = self.df_bin.shape[0]

    def generate(self):

        # Domain prediction columns
        pred_cols = [i for i in self.df.columns if 'Domain Prediction' in i]

        # Miscalibration area vs. RMSE
        area_vs_rmse(self.df_bin, self.save)

        # Loop over domains
        for i, j, k, f, z in zip(
                                 self.domains,
                                 self.errors,
                                 self.assessments,
                                 self.gts,
                                 self.prs,
                                 ):

            # Separate domains and classes
            for group, df in self.df.groupby(i):

                # Parity plot
                parity(
                       df.y,
                       df.y_pred,
                       df.y_stdc_pred,
                       df['r/std_y'],
                       df.d_pred,
                       self.save,
                       '{}_{}'.format(k, group),
                       )

                # Need to re-bin data by stdc not d for visual
                df = bin_data(df, self.bins, 'y_stdc_pred/std_y')

                # RMSE vs. stdc
                rmse_vs_stdc(
                             df,
                             self.save,
                             '{}_{}'.format(k, group),
                             )

            # CDF Plots
            cdf(self.df, i, self.save, k)

            # Bins versus errors
            bins(
                 self.df,
                 'd_pred',
                 j,
                 i,
                 f,
                 r'$E^{{{}}}$'.format(k),
                 self.save,
                 k,
                 )

            # PR curve
            pr(z, self.save, k)

            # Confusion matrices
            for pred in pred_cols:
                if i.replace('domain_', '') in pred:
                    y = self.df.loc[:, i].values
                    y_pred = self.df.loc[:, pred].values
                    suffix = pred.replace(' ', '_').replace('/', '_div_')
                    confusion(y, y_pred, self.save, suffix)
