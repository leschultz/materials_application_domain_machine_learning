from sklearn.metrics import (
                             PrecisionRecallDisplay,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             )

from matplotlib import pyplot as pl
from madml.utils import parallel
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
                                        ), bbox_inches='tight', dpi=400)

        ax.legend([]).set_visible(False)

        pl.close(fig_legend)

    fig.savefig(os.path.join(
                             save,
                             '{}_{}.png'.format(name, suffix)
                             ), bbox_inches='tight', dpi=400)

    pl.close(fig)

    jsonfile = os.path.join(save, '{}_{}.json'.format(name, suffix))
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def residuals(df, save='.', suffix='d'):
    '''
    A plot of absolute residuals vs. dissimilarity.

    inputs:
        df = Data.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    data = {}
    fig, ax = pl.subplots()

    x = df['d_pred'].values
    y = df['r/std_y'].abs().values

    ax.scatter(x, y, marker='.', color='r')

    data['d_pred'] = x.tolist()
    data['r/std_y'] = y.tolist()

    ax.set_xlabel(r'$d$')
    ax.set_ylabel(r'$|y-\hat{y}|/\sigma_{y}$')

    plot_dump(data, fig, ax, 'residuals', save, suffix, False)


def confidence(df, save='.', suffix='all'):
    '''
    A plot of absolute residuals vs. dissimilarity.

    inputs:
        df = Data.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    def func(i, y, z):
        y_sub = y[i:]
        z_sub = z[i:]

        rmse = (sum(y_sub**2)/y_sub.shape[0])**0.5
        area = calculators.cdf(z_sub)[-1]

        return rmse, area

    sorters = {
               r'$|y-\hat{y}|/\sigma_{y}$': df['r/std_y'].abs().values,
               r'$d$': df['d_pred'].values,
               r'$\sigma_{c}$': df['y_stdc_pred'].values,
               'Random': np.random.uniform(size=df.shape[0]),
               }

    loop = range(0, df.shape[0])
    frac = list(np.array(loop)/df.shape[0])

    data_area = {}
    data_rmse = {}
    fig_rmse, ax_rmse = pl.subplots()
    fig_area, ax_area = pl.subplots()
    for key, value in sorters.items():

        indx = np.argsort(value)[::-1]
        y = sorters[r'$|y-\hat{y}|/\sigma_{y}$'][indx]
        z = df['z'].values[indx]

        out = parallel(
                       func,
                       loop,
                       y=y,
                       z=z,
                       disable=True,
                       )

        rmse = [i[0] for i in out]
        area = [i[1] for i in out]

        auc_rmse = np.trapz(rmse, x=frac, dx=0.00001)
        auc_area = np.trapz(rmse, x=frac, dx=0.00001)

        label_rmse = '{} AUC: {:.2f}'.format(key, auc_rmse)
        label_area = '{} AUC: {:.2f}'.format(key, auc_area)

        ax_rmse.plot(frac, rmse, label=label_rmse)
        ax_area.plot(frac, area, label=label_area)

        data_rmse[key] = {'x': frac, 'y': rmse, 'auc': auc_rmse}
        data_area[key] = {'x': frac, 'y': area, 'auc': auc_area}

    ax_rmse.set_xlabel('Fraction Removed')
    ax_rmse.set_ylabel(r'$E^{RMSE/\sigma_{y}}$')

    ax_area.set_xlabel('Fraction Removed')
    ax_area.set_ylabel(r'$E^{area}$')

    plot_dump(data_rmse, fig_rmse, ax_rmse, 'confidence_rmse', save, suffix)
    plot_dump(data_area, fig_area, ax_area, 'confidence_area', save, suffix)


def parity(
           df,
           save='.',
           suffix='',
           heat=False,
           ):
    '''
    Make a parity plot.

    inputs:
        df = Data.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    y = df.y
    y_pred = df.y_pred
    y_stdc_pred = df.y_stdc_pred
    r_std_y = df['r/std_y']
    d = df.d_pred

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

    if heat:
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

        fig.colorbar(sc, ax=ax, label=r'$d$')

    else:
        sc = ax.scatter(
                        y,
                        y_pred,
                        marker='.',
                        zorder=2,
                        label=label,
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
        gt = The column to group.
        save = The location to save the figure/data.
    '''

    data = {}
    fig, ax = pl.subplots()
    for group, values in df.groupby(gt):

        eval_points, y, y_pred, areacdf = calculators.cdf(values['z'])

        area_label = '{}: '.format(group)
        area_label += '$E^{{area}}={:.3f}$'.format(areacdf)

        color = 'r' if group == 'OD' else 'g'

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
            label=r'$\Phi(0,1)$',
            )

    data[r'$\Phi(0,1)$'] = {}
    data[r'$\Phi(0,1)$']['x'] = list(eval_points)
    data[r'$\Phi(0,1)$']['y'] = list(y)
    data[r'$\Phi(0,1)$'] = areacdf

    ax.set_xlabel('z')
    ax.set_ylabel('CDF(z)')

    plot_dump(data, fig, ax, 'cdf', save, suffix)


def bins(df, d, e, elabel, gt, ylabel, gtlabel, save, suffix):
    '''
    Plot statistical errors with respect to dissimilarity.

    inputs:
        d = The dissimilarity.
        e = The error statistic.
        elabel = The domain labels.
        gt = The domain ground truth.
        ylabel = The y-axis label.
        gtlabel = The label for ground truth.
        save = The directory to save plot.
        suffix = Append a suffix to the save name.
    '''

    # Range of ground truths from training.
    gt_min = float(df[gt].min())  # Should now be all the same

    p = {
         'ID': {
                'color': 'g',
                'marker': '.',
                'zorder': 2,
                },
         'OD': {
                'color': 'r',
                'marker': 'x',
                'zorder': 1,
                },
         }

    data = {'gt_min': gt_min}
    data['ID'] = {'x': [], 'y': []}
    data['OD'] = {'x': [], 'y': []}

    fig, ax = pl.subplots()
    for group, values in df.groupby([elabel, 'bin'], observed=True):

        dom, _ = group

        x = np.array(values[d], dtype=float)
        y = np.array(values[e], dtype=float)

        ax.scatter(
                   x,
                   y,
                   alpha=0.5,
                   **p[dom],
                   )

        ax.fill_between(
                        x,
                        y,
                        color=p[dom]['color'],
                        alpha=0.5,
                        )

        data[dom]['x'].append(x.tolist())
        data[dom]['y'].append(y.tolist())

    for key, val in p.items():
        ax.scatter([], [], label=key, **val)

    ax.axhline(
               gt_min,
               color='g',
               label='{} = {:.2f}'.format(gtlabel, gt_min),
               )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$d$')

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

    f1_id = metrics.f1_score(y, y_pred, pos_label='ID')
    f1_od = metrics.f1_score(y, y_pred, pos_label='OD')
    conf = confusion_matrix(y, y_pred, labels=['ID', 'OD'], normalize='all')

    fig, ax = pl.subplots()
    disp = ConfusionMatrixDisplay(
                                  conf,
                                  display_labels=['ID', 'OD'],
                                  )
    disp.plot(ax=ax)

    data = {}
    data['confusion_matrix'] = conf.tolist()
    data['F1_ID'] = f1_id
    data['F1_OD'] = f1_od

    plot_dump(data, fig, ax, 'confusion', save, suffix, legend=False)


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

    fig.colorbar(sc, ax=ax, label=r'$d$')

    ax.set_xlabel(r'$E^{RMSE/\sigma_{y}}$')
    ax.set_ylabel(r'$E^{area}$')

    data = {}
    data['x'] = rmse.tolist()
    data['y'] = area.tolist()
    data['d'] = d.tolist()

    plot_dump(data, fig, ax, 'area_vs_RMSE', save, 'vs_d')


def rmse_vs_stdc(df, save, suffix, heat=False):
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
    if heat:
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

        fig.colorbar(sc, ax=ax, label=r'$d$')

    else:
        sc = ax.scatter(
                        stdc,
                        rmse,
                        marker='.',
                        zorder=2,
                        label='Bins',
                        )

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
            label=r'$E^{RMSE/\sigma_{y}}=mean(\sigma_{c}/\sigma_{y})$',
            color='k',
            linestyle=':',
            zorder=1
            )

    ax.set_aspect('equal')

    ax.set_xlabel(r'$\sigma_{c}/\sigma(y)$')
    ax.set_ylabel(r'$E^{RMSE/\sigma_{y}}$')

    data = {}
    data['x'] = stdc.tolist()
    data['y'] = rmse.tolist()
    data['d'] = d.tolist()

    plot_dump(data, fig, ax, 'RMSE_vs_stdc', save, suffix)


class plotter:

    def __init__(
                 self,
                 df,
                 df_bin,
                 df_confusion=None,
                 precs=None,
                 save='.',
                 ):

        self.save = save
        self.domains = ['domain_rmse/std_y', 'domain_cdf_area']
        self.errors = ['rmse/std_y', 'cdf_area']
        self.gts = ['gt_rmse', 'gt_area']
        self.assessments = ['rmse', 'area']
        self.precs = precs  # Precisions used

        # For plotting purposes on the histogram of E^* vs. D
        cols = self.errors+self.domains
        self.df = df.sort_values(by=['d_pred']+cols)
        self.df_bin = df_bin.sort_values(by=['d_pred_max']+cols)

        self.bins = self.df_bin.shape[0]

        # Domain prediction columns
        pred_cols = [i for i in self.df.columns if 'Domain Prediction' in i]

        if df_confusion is None:
            self.df_confusion = df[self.domains+pred_cols+['splitter']]
            self.pred_cols = pred_cols
        else:
            self.df_confusion = df_confusion
            self.pred_cols = [i.split(' (')[0] for i in pred_cols]

    def generate(self):

        # Write test data
        self.df.to_csv(os.path.join(self.save, 'pred.csv'), index=False)
        self.df_bin.to_csv(os.path.join(
                                        self.save,
                                        'pred_bins.csv',
                                        ), index=False)

        # For data used to fit regression model
        df = self.df[self.df['splitter'] == 'fit']
        parity(
               df,
               self.save,
               'fit_splitter',
               )

        # Residuals
        residuals(self.df, self.save)

        # Confidence
        confidence(self.df, self.save)
        confidence(df, self.save, 'fit_splitter')

        # CDF
        cdf(df, 'splitter', self.save, 'fit_splitter')

        # Need to re-bin data by stdc not d for visual
        _, df = calculators.bin_data(df, self.bins, 'y_stdc_pred/std_y')

        # RMSE vs. stdc
        rmse_vs_stdc(
                     df,
                     self.save,
                     'fit_splitter',
                     )

        # Miscalibration area vs. RMSE
        area_vs_rmse(self.df_bin, self.save)

        # Prepare for confusion matrixes
        dy = self.df_confusion['splitter'] == 'fit'
        dy = self.df_confusion[dy]
        dn = self.df_confusion['splitter'] != 'fit'
        dn = self.df_confusion[dn]

        # Loop over domains
        for i, j, k, f in zip(
                              self.domains,
                              self.errors,
                              self.assessments,
                              self.gts,
                              ):

            # Name change Dane wants
            if k == 'rmse':
                ename = r'$E^{RMSE/\sigma_{y}}$'
                cname = r'$E^{RMSE/\sigma_{y}}_{c}$'
            elif k == 'area':
                ename = r'$E^{area}$'
                cname = r'$E^{area}_{c}$'
            else:
                raise 'Unsupported error metric'

            # Separate domains and classes
            for group, df in self.df.groupby(i):

                # Parity plot
                parity(
                       df,
                       self.save,
                       '{}_{}'.format(k, group),
                       )

                # Need to re-bin data by stdc not d for visual
                _, df = calculators.bin_data(
                                             df,
                                             self.bins,
                                             'y_stdc_pred/std_y',
                                             )

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
                 ename,
                 cname,
                 self.save,
                 k,
                 )

            # PR curve
            pr_data = calculators.pr(
                                     self.df['d_pred'],
                                     self.df[i],
                                     self.precs,
                                     )
            pr(pr_data, self.save, k)

            # Confusion matrices
            for pred in self.pred_cols:
                if i.replace('domain_', '') in pred:

                    # Confusion matrix for all splitters
                    y = self.df_confusion.loc[:, i].values
                    y_pred = self.df_confusion.loc[:, pred].values
                    suffix = pred.replace(' ', '_').replace('/', '_div_')
                    confusion(y, y_pred, self.save, suffix+'_splitter_all')

                    # Confusion matrix for fit splitters
                    y = dy.loc[:, i].values
                    y_pred = dy.loc[:, pred].values
                    suffix = pred.replace(' ', '_').replace('/', '_div_')
                    confusion(y, y_pred, self.save, suffix+'_splitter_fit')

                    # Confusion matrix for spliters that are not fit
                    y = dn.loc[:, i].values
                    y_pred = dn.loc[:, pred].values
                    suffix = pred.replace(' ', '_').replace('/', '_div_')
                    confusion(y, y_pred, self.save, suffix+'_splitter_not_fit')
