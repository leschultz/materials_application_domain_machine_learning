from sklearn.metrics import (
                             precision_recall_curve,
                             ConfusionMatrixDisplay,
                             confusion_matrix,
                             auc
                             )

from matplotlib import pyplot as pl
from scipy import stats

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib
import json
import os

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def parity(
           mets,
           y,
           y_pred,
           in_domain,
           y_pred_sem=None,
           name='',
           units='',
           save='.'
           ):
    '''
    Make a paroody plot.

    inputs:
        mets = The regression metrics.
        y = The true target value.
        y_pred = The predicted target value.
        y_pred_sem = The standard error of the mean in predicted values.
        name = The name of the target value.
        units = The units of the target value.
        save = The directory to save plot.
    '''

    os.makedirs(save, exist_ok=True)

    out_domain = ~in_domain

    labels = {}
    mets_save = {}

    if y_pred_sem is not None:

        for i in [True, False]:

            m = mets[mets['in_domain'] == i]

            if m.shape[0] > 0:

                m = m.to_dict(orient='records')[0]

                rmse_sigma = m[r'$RMSE/\sigma_{y}$_mean']
                rmse_sigma_sem = m[r'$RMSE/\sigma_{y}$_sem']

                rmse = m[r'$RMSE$_mean']
                rmse_sem = m[r'$RMSE$_sem']

                mae = m[r'$MAE$_mean']
                mae_sem = m[r'$MAE$_sem']

                r2 = m[r'$R^{2}$_mean']
                r2_sem = m[r'$R^{2}$_sem']

                label = r'$RMSE/\sigma_{y}=$'
                label += r'{:.2} $\pm$ {:.2}'.format(
                                                     rmse_sigma,
                                                     rmse_sigma_sem
                                                     )
                label += '\n'
                label += r'$RMSE=$'
                label += r'{:.2} $\pm$ {:.2}'.format(rmse, rmse_sem)
                label += '\n'
                label += r'$MAE=$'
                label += r'{:.2} $\pm$ {:.2}'.format(mae, mae_sem)
                label += '\n'
                label += r'$R^{2}=$'
                label += r'{:.2} $\pm$ {:.2}'.format(r2, r2_sem)

                labels[i] = label
                mets_save[i] = m

            else:
                labels[i] = 'No out of domain'
                mets_save[i] = 'No out of domain'

    else:

        for i in [True, False]:

            m = mets[mets['in_domain'] == i]

            if m.shape[0] > 0:

                m = m.to_dict(orient='records')[0]

                rmse_sigma = m[r'$RMSE/\sigma_{y}$']
                rmse = m[r'$RMSE$']
                mae = m[r'$MAE$']
                r2 = m[r'$R^{2}$']

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

                labels[i] = label
                mets_save[i] = m

            else:
                labels[i] = 'No out of domain'
                mets_save[i] = 'No out of domain'

    fig, ax = pl.subplots()

    if y_pred_sem is not None:
        ax.errorbar(
                    y[in_domain],
                    y_pred[in_domain],
                    y_pred_sem[in_domain],
                    linestyle='none',
                    marker='.',
                    markerfacecolor='None',
                    zorder=1,
                    color='b',
                    )
        ax.errorbar(
                    y[out_domain],
                    y_pred[out_domain],
                    yerr=y_pred_sem[out_domain],
                    linestyle='none',
                    marker='x',
                    markerfacecolor='None',
                    zorder=0,
                    color='r',
                    )

    ax.scatter(
               y[in_domain],
               y_pred[in_domain],
               marker='.',
               zorder=2,
               color='b',
               label=labels[True],
               )

    ax.scatter(
               y[out_domain],
               y_pred[out_domain],
               marker='x',
               zorder=1,
               color='r',
               label=labels[False],
               )

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

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel('Predicted {} {}'.format(name, units))
    ax.set_xlabel('Actual {} {}'.format(name, units))

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    fig.savefig(os.path.join(save, 'parity.png'), bbox_inches='tight')
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['y_pred_id'] = list(y_pred[in_domain])
    data['y_id'] = list(y[in_domain])
    data['metrics'] = mets_save[True]

    data['y_pred_od'] = list(y_pred[out_domain])
    data['y_od'] = list(y[out_domain])
    data['metrics'] = mets_save[False]

    if y_pred_sem is not None:
        data['y_pred_sem'] = list(y_pred_sem[in_domain])
        data['y_pred_sem'] = list(y_pred_sem[out_domain])

    jsonfile = os.path.join(save, 'parity.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def cdf(x):
    '''
    Plot the quantile quantile plot for cummulative distributions.
    inputs:
        x = The residuals normalized by the calibrated uncertainties.
    '''

    nx = len(x)
    nz = 100000
    z = np.random.normal(0, 1, nz)  # Standard normal distribution

    # Need sorting
    x = sorted(x)
    z = sorted(z)

    # Cummulative fractions
    xfrac = np.arange(nx)/(nx-1)
    zfrac = np.arange(nz)/(nz-1)

    # Interpolation to compare cdf
    eval_points = sorted(list(set(x+z)))
    y_pred = np.interp(eval_points, x, xfrac)  # Predicted
    y = np.interp(eval_points, z, zfrac)  # Standard Normal

    # Area bertween ideal Gaussian and observed
    area = np.trapz(abs(y_pred-y), x=y, dx=0.00001)

    return y, y_pred, area


def cdf_parity(x, in_domain, save):
    '''
    Plot the quantile quantile plot for cummulative distributions.
    inputs:
        x = The residuals normalized by the calibrated uncertainties.
    '''

    os.makedirs(save, exist_ok=True)

    out_domain = ~in_domain

    data = {}
    fig, ax = pl.subplots()

    y, y_pred, area = cdf(x)
    ax.plot(
            y,
            y_pred,
            zorder=0,
            color='b',
            label='Total Area: {:.3f}'.format(area),
            )
    data['y'] = list(y)
    data['y_pred'] = list(y_pred)

    if x[in_domain].shape[0] > 1:
        y_id, y_pred_id, in_area = cdf(x[in_domain])
        ax.plot(
                y_id,
                y_pred_id,
                zorder=0,
                color='g',
                label='ID Area: {:.3f}'.format(in_area),
                )
        data['y_id'] = list(y_id)
        data['y_pred_id'] = list(y_pred_id)

    if x[out_domain].shape[0] > 1:
        y_od, y_pred_od, out_area = cdf(x[out_domain])
        ax.plot(
                y_od,
                y_pred_od,
                zorder=0,
                color='r',
                label='OD Area: {:.3f}'.format(out_area),
                )
        data['y_od'] = list(y_od)
        data['y_pred_od'] = list(y_pred_od)

    # Line of best fit
    ax.plot(
            [0, 1],
            [0, 1],
            color='k',
            linestyle=':',
            zorder=1,
            )

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_ylabel('Predicted CDF')
    ax.set_xlabel('Standard Normal CDF')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    ax.set_aspect('equal')
    fig.savefig(os.path.join(save, 'cdf_parity.png'), bbox_inches='tight')

    pl.close(fig)

    jsonfile = os.path.join(save, 'cdf_parity.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def intervals(df, metric, save=False):
    '''
    Plot the confidence curve:
    '''

    gt = 0.01
    q = 10
    metric_name = metric+'_bin'

    df['absres'] = abs(df['y'].values-df['y_pred'].values)
    df['absres'] = df['absres']/df['sigma_y']

    df['bin'] = pd.qcut(df[metric].rank(method='first'), q=q)

    groups = df.groupby('bin')

    zvars = []
    mdists = []
    mdists_mins = []
    mdists_maxs = []
    rmses = []
    stds = []
    pvals = []
    for group, values in groups:

        rmse = values['absres'].values
        rmse = rmse**2
        rmse = sum(rmse)/len(rmse)
        rmse = rmse**0.5

        zvar = values['z'].var()

        m = values[metric]
        mins = m.min()
        maxs = m.max()
        m = m.mean()

        if values.shape[0] < 2:
            pval = 0.0
        else:
            pval = stats.cramervonmises(values['z'], 'norm').pvalue

        df.loc[df['bin'] == group, metric_name] = maxs
        df.loc[df['bin'] == group, 'pval'] = pval

        zvars.append(zvar)
        mdists.append(m)
        mdists_mins.append(mins)
        mdists_maxs.append(maxs)
        rmses.append(rmse)
        pvals.append(pval)

    pvals = np.array(pvals)
    in_domain = pvals > gt
    out_domain = ~in_domain

    df['in_domain_'+metric_name] = df['pval'] > gt

    if save is not False:

        os.makedirs(save, exist_ok=True)

        avg_points = df.shape[0]/q
        zvartot = df['z'].var()

        if 'y_std' in metric:
            xlabel = 'Mean $\sigma_{c}/\sigma_{y}$'
        else:
            xlabel = 'Mean Dissimilarity'

        fig, ax = pl.subplots()

        mdists = np.array(mdists)
        zvars = np.array(zvars)
        rmses = np.array(rmses)
        pointlabel = 'PPB = {:.2f}'.format(avg_points)

        ax.scatter(
                   mdists[in_domain],
                   zvars[in_domain],
                   marker='.',
                   color='b',
                   label='ID '+pointlabel,
                   )

        ax.scatter(
                   mdists[out_domain],
                   zvars[out_domain],
                   marker='x',
                   color='b',
                   label='OD '+pointlabel,
                   )

        ax.scatter(
                   mdists_mins,
                   zvars,
                   marker='|',
                   color='r',
                   label='Bin Start',
                   )

        ax.scatter(
                   mdists_maxs,
                   zvars,
                   marker='|',
                   color='k',
                   label='Bin End',
                   )

        ax.axhline(1.0, color='r', label='Ideal VAR(z) = 1.0')
        ax.axhline(zvartot, label='Total VAR(z) = {:.1f}'.format(zvartot))

        ax.set_ylim(0.0, None)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'VAR(z)')

        fig.savefig(os.path.join(
                                 save,
                                 'confidence.png'
                                 ), bbox_inches='tight')

        pl.close(fig)

        data = {}
        data['x_id'] = mdists[in_domain].tolist()
        data['y_id'] = zvars[in_domain].tolist()
        data['x_od'] = mdists[out_domain].tolist()
        data['y_od'] = zvars[out_domain].tolist()
        data['x_min'] = mdists_mins
        data['x_max'] = mdists_maxs
        data['ppb'] = avg_points
        data['z_var_total'] = zvartot

        jsonfile = os.path.join(save, 'confidence.json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)

        fig, ax = pl.subplots()

        ax.scatter(
                   mdists[in_domain],
                   rmses[in_domain],
                   marker='.',
                   color='b',
                   label='ID '+pointlabel,
                   )

        ax.scatter(
                   mdists[out_domain],
                   rmses[out_domain],
                   marker='x',
                   color='b',
                   label='OD '+pointlabel,
                   )

        ax.scatter(
                   mdists_mins,
                   rmses,
                   marker='|',
                   color='r',
                   label='Bin Start',
                   )

        ax.scatter(
                   mdists_maxs,
                   rmses,
                   marker='|',
                   color='k',
                   label='Bin End',
                   )

        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x, linestyle=':', color='k', label='Ideal')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('$RMSE/\sigma_{y}$')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.savefig(os.path.join(
                                 save,
                                 'rmse_vs_uq.png'
                                 ), bbox_inches='tight')

        pl.close(fig)

        data['x_id'] = mdists[in_domain].tolist()
        data['y_id'] = rmses[in_domain].tolist()
        data['x_od'] = mdists[out_domain].tolist()
        data['y_od'] = rmses[out_domain].tolist()
        data['x_min'] = mdists_mins
        data['x_max'] = mdists_maxs
        data['ppb'] = avg_points
        data['z_var_total'] = zvartot

        jsonfile = os.path.join(save, 'rmse_vs_uq.json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)

        fig, ax = pl.subplots()

        ax.scatter(
                   mdists[in_domain],
                   pvals[in_domain],
                   marker='.',
                   color='b',
                   label='ID '+pointlabel,
                   )

        ax.scatter(
                   mdists[out_domain],
                   pvals[out_domain],
                   marker='x',
                   color='b',
                   label='OD '+pointlabel,
                   )

        ax.scatter(
                   mdists_mins,
                   pvals,
                   marker='|',
                   color='r',
                   label='Bin Start',
                   )

        ax.scatter(
                   mdists_maxs,
                   pvals,
                   marker='|',
                   color='k',
                   label='Bin End',
                   )

        ax.axhline(
                   gt,
                   color='k',
                   linestyle=':',
                   label='GT = {:.2f}'.format(gt),
                   )

        ax.set_yscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('p-value')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.savefig(os.path.join(
                                 save,
                                 'p_vs_uq.png'
                                 ), bbox_inches='tight')

        pl.close(fig)

        data['x_id'] = mdists[in_domain].tolist()
        data['y_id'] = pvals[in_domain].tolist()
        data['x_od'] = mdists[out_domain].tolist()
        data['y_od'] = pvals[out_domain].tolist()
        data['x_min'] = mdists_mins
        data['x_max'] = mdists_maxs
        data['ppb'] = avg_points

        jsonfile = os.path.join(save, 'p_vs_uq.json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)

    return df


def ground_truth(y, y_pred, y_std, std, in_domain, save):

    os.makedirs(save, exist_ok=True)

    absres = abs(y-y_pred)/std
    y_std = y_std/std

    out_domain = ~in_domain

    fig, ax = pl.subplots()

    ax.scatter(absres[in_domain], y_std[in_domain], color='g', marker='.')
    ax.scatter(absres[out_domain], y_std[out_domain], color='r', marker='x')

    ax.set_xlabel(r'$|y-\hat{y}|/\sigma_{y}$')
    ax.set_ylabel(r'$\sigma_{c}/\sigma_{y}$')

    fig.savefig(os.path.join(save, 'ground_truth.png'), bbox_inches='tight')
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['x_green'] = list(absres[in_domain])
    data['y_green'] = list(y_std[in_domain])
    data['x_red'] = list(absres[out_domain])
    data['y_red'] = list(y_std[out_domain])

    jsonfile = os.path.join(save, 'ground_truth.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def violin(dist, in_domain, save):
    '''
    Plot the violin plot.
    '''

    df = {'dist': dist, 'in_domain': in_domain}
    df = pd.DataFrame(df)

    groups = df.groupby('in_domain')
    median = groups.median()
    dist = median['dist'].sort_values(ascending=False)
    dist = dist.to_frame().reset_index()['in_domain'].values

    df['in_domain'] = pd.Categorical(df['in_domain'], dist)

    fig, ax = pl.subplots()
    sns.violinplot(
                   data=df,
                   x='dist',
                   y='in_domain',
                   ax=ax,
                   palette='Spectral',
                   cut=0,
                   scale='width',
                   inner='quartile'
                   )

    fig.savefig(os.path.join(save, 'violin.png'), bbox_inches='tight')
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['x'] = list(df['dist'].values)
    data['y'] = list(df['in_domain'].values)

    jsonfile = os.path.join(save, 'violin.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def assessment(
               y_res,
               std,
               dist,
               in_domain,
               save,
               thresh=None
               ):

    y_res_norm = y_res/std
    os.makedirs(save, exist_ok=True)

    out_domain = ~in_domain

    slope, intercept, r, p, se = stats.linregress(dist, y_res_norm)

    fig, ax = pl.subplots()

    ax.scatter(dist[in_domain], y_res_norm[in_domain], color='g', marker='.')
    ax.scatter(dist[out_domain], y_res_norm[out_domain], color='r', marker='x')

    xfit = np.linspace(min(dist), max(dist))
    yfit = slope*xfit+intercept

    ax.plot(
            xfit,
            yfit,
            color='k',
            label='Slope: {:.2f}\nIntercept: {:.2f}'.format(slope, intercept)
            )

    if thresh:
        ax.axvline(thresh, color='k')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xlabel('Dissimilarity')
    ax.set_ylabel(r'$|y-\hat{y}|/\sigma_{y}$')

    fig.savefig(os.path.join(save, 'assessment.png'), bbox_inches='tight')
    pl.close(fig)

    # Repare plot data for saving
    data = {}
    data['x_green'] = list(dist[in_domain])
    data['y_green'] = list(y_res_norm[in_domain])
    data['x_red'] = list(dist[out_domain])
    data['y_red'] = list(y_res_norm[out_domain])

    if thresh:
        data['vertical'] = thresh

    jsonfile = os.path.join(save, 'assessment.json')
    with open(jsonfile, 'w') as handle:
        json.dump(data, handle)


def pr(score, in_domain, pos_label, save=False):

    if pos_label is True:
        score = -score

    baseline = [1 if i == pos_label else 0 for i in in_domain]
    baseline = sum(baseline)/len(in_domain)
    relative_base = 1-baseline  # The amount of area to gain in PR

    precision, recall, thresholds = precision_recall_curve(
                                                           in_domain,
                                                           score,
                                                           pos_label=pos_label,
                                                           )

    num = 2*recall*precision
    den = recall+precision
    f1_scores = np.divide(
                          num,
                          den,
                          out=np.zeros_like(den), where=(den != 0)
                          )

    # Maximum F1 score
    max_f1_index = np.argmax(f1_scores)

    # AUC score
    auc_score = auc(recall, precision)
    auc_relative = (auc_score-baseline)/relative_base

    # Relative f1
    precision_rel = precision-baseline
    precision_rel = precision_rel.clip(min=0.0)
    num = 2*recall*precision_rel
    den = recall+precision_rel
    f1_rel = np.divide(
                       num,
                       den,
                       out=np.zeros_like(den), where=(den != 0)
                       )

    # Maximum Relative F1 score
    rel_f1_index = np.argmax(f1_rel)

    custom = {}
    custom['Max F1'] = {
                        'Precision': precision[max_f1_index],
                        'Recall': recall[max_f1_index],
                        'Threshold': thresholds[max_f1_index],
                        'F1': f1_scores[max_f1_index],
                        }

    custom['Max Relative F1'] = {
                                 'Precision': precision[rel_f1_index],
                                 'Recall': recall[rel_f1_index],
                                 'Threshold': thresholds[rel_f1_index],
                                 'F1': f1_scores[rel_f1_index],
                                 }

    # Loop for lowest to highest to get better thresholds than the other way
    nprec = len(precision)
    nthresh = nprec-1  # sklearn convention
    nthreshindex = nthresh-1  # Foor loop index comparison
    loop = range(nprec)
    for cut in [0.95, 0.75, 0.5, 0.25]:

        for index in loop:
            p = precision[index]
            if p >= cut:
                break

        name = 'Minimum Precision: {}'.format(cut)
        custom[name] = {
                        'Precision': precision[index],
                        'Recall': recall[index],
                        'F1': f1_scores[index],
                        }

        # If precision is set at arbitrary 1 from sklearn convetion
        if index > nthreshindex:
            custom[name]['Threshold'] = max(thresholds)
        else:
            custom[name]['Threshold'] = thresholds[index]

    # Convert back
    if pos_label is True:
        thresholds = -thresholds
        for key, value in custom.items():
            custom[key]['Threshold'] *= -1

    if save is not False:

        os.makedirs(save, exist_ok=True)

        fig, ax = pl.subplots()

        ax.plot(
                recall,
                precision,
                color='b',
                label='AUC: {:.2f}\nRelative AUC: {:.2f}'.format(
                                                                 auc_score,
                                                                 auc_relative
                                                                 ),
                )
        ax.hlines(
                  baseline,
                  color='r',
                  linestyle=':',
                  label='Baseline: {:.2f}'.format(baseline),
                  xmin=0.0,
                  xmax=1.0,
                  )

        for key, values in custom.items():
            p = custom[key]['Precision']
            r = custom[key]['Recall']
            t = custom[key]['Threshold']

            label = key
            label += '\nPrecision: {:.2f}'.format(p)
            label += '\nRecall: {:.2f}'.format(r)
            label += '\nThreshold: {:.2f}'.format(t)
            ax.scatter(r, p, marker='o', label=label)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        fig.savefig(os.path.join(save, 'pr.png'), bbox_inches='tight')
        pl.close(fig)

        # Repare plot data for saving
        data = {}
        data['recall'] = list(recall)
        data['precision'] = list(precision)
        data['baseline'] = baseline
        data['auc'] = auc_score
        data['auc_relative'] = auc_relative
        data.update(custom)

        jsonfile = os.path.join(save, 'pr.json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)

        fig, ax = pl.subplots()

        ax.plot(
                thresholds,
                recall[:-1],
                color='b',
                label='Recall'
                )

        ax.plot(
                thresholds,
                precision[:-1],
                color='k',
                label='Precision'
                )

        ax.plot(
                thresholds,
                f1_scores[:-1],
                color='g',
                label='F1'
                )

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_ylim(0.0, 1.05)

        ax.set_xlabel('Thresholds')
        ax.set_ylabel('Recall, Precision, or F1')

        fig.savefig(os.path.join(save, 'thresholds.png'), bbox_inches='tight')
        pl.close(fig)

        # Repare plot data for saving
        data = {}
        data['recall'] = list(recall[:-1])
        data['precision'] = list(precision[:-1])
        data['thresholds'] = list(thresholds)

        jsonfile = os.path.join(save, 'thresholds.json')
        with open(jsonfile, 'w') as handle:
            json.dump(data, handle)

    return custom


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
