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

np.random.seed(0)

# Standard normal distribution
nz = 10000
z_standard_normal = np.random.normal(0, 1, nz)


def generate_plots(data_cv, ystd, bins, save, gts, gtb, dists):
    '''
    A function that standardizes plot generation.

    inputs:
        data_cv = The cross validation data.
        ystd = The standard deviation of the target variable.
        bins = The number of bins for data.
        save = The location to save plots.

    outputs:
        th = The thresholds from PR curve.
        data_cv_bin = The binned data.
    '''

    uqcond = 'z' in data_cv.columns  # Condition to do UQ
    dscond = 'dist' in data_cv.columns  # Condition for distance

    th = {}
    data_cv_bin = {}
    if save:

        singlesave = os.path.join(save, 'single')

        parity(
               data_cv['y'].values,
               data_cv['y_pred'].values,
               ystd,
               singlesave,
               'total',
               )

        if uqcond:
            intervalsave = os.path.join(save, 'intervals')

            cdf(
                data_cv['z'],
                intervalsave,
                subsave='_total',
                )

        # For each splitter of data
        for split, values in data_cv.groupby('splitter'):

            sub = '{}'.format(split)
            parity(
                   values['y'].values,
                   values['y_pred'].values,
                   ystd,
                   singlesave,
                   sub,
                   )
            if uqcond:
                cdf(
                    values['z'],
                    intervalsave,
                    subsave='_'+sub
                    )

    else:
        singlesave = intervalsave = save

    for i in dists:

        if save:
            name = i.replace('/', '_')
            singledistsave = os.path.join(singlesave, name)

            if uqcond:
                intervaldistsave = os.path.join(
                                                intervalsave,
                                                name,
                                                )
        else:
            singledistsave = intervaldistsave = save

        if singledistsave:
            single_truth(data_cv, i, singledistsave, gts)

        if uqcond:

            dist_bin = binned_truth(
                                    data_cv,
                                    i,
                                    bins,
                                    gtb,
                                    save=intervaldistsave,
                                    )
            data_cv_bin[i] = dist_bin

        th[i] = {}
        for j, k in zip([True, False], ['id', 'od']):

            if save:
                singledomainsave = os.path.join(singledistsave, k)

                if uqcond:
                    intervaldomainsave = os.path.join(intervaldistsave, k)
            else:
                singledomainsave = intervaldomainsave = save

            thresh = pr(
                        data_cv[i].values,
                        data_cv['id'].values,
                        j,
                        save=singledomainsave,
                        )

            th[i][k] = thresh

            if uqcond:
                thresh_bin = pr(
                                data_cv_bin[i][i+'_max'].values,
                                data_cv_bin[i]['id'].values,
                                j,
                                save=intervaldomainsave,
                                )

                th[i][k+'_bin'] = thresh_bin

    return th, data_cv_bin


def parity(
           y,
           y_pred,
           sigma_y,
           save='.',
           subname='',
           ):
    '''
    Make a paroody plot.

    inputs:
        mets = The regression metrics.
        y = The true target value.
        y_pred = The predicted target value.
        sigma_y = The standard deviation of target variable.
        save = The directory to save plot.
    '''

    os.makedirs(save, exist_ok=True)

    rmse = metrics.mean_squared_error(y, y_pred)**0.5

    if y.shape[0] > 1:
        rmse_sigma = rmse/sigma_y
    else:
        rmse_sigma = np.nan

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

    ax.scatter(
               y,
               y_pred,
               marker='.',
               zorder=2,
               color='b',
               label=label,
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
                             'parity_{}.png'.format(subname)
                             ), bbox_inches='tight')
    fig_legend.savefig(os.path.join(
                                    save,
                                    'parity_{}_legend.png'.format(subname)
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

    jsonfile = os.path.join(save, 'parity_{}.json'.format(subname))
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

    nx = len(x)

    if choice == 'standard_normal':
        z = z_standard_normal
    elif choice == 'same_mean':
        z = np.random.normal(np.mean(x), 1, nz)  # Shifted mean
    elif choice == 'same_variance':
        z = np.random.normal(0, np.var(x), nz)  # Altered variance

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

    # Cramer–von Mises test
    pvalue = stats.cramervonmises(x, 'norm').pvalue

    # Area between ideal distribution and observed
    absres = np.abs(y_pred-y)
    areacdf = np.trapz(absres, x=eval_points, dx=0.00001)
    areaparity = np.trapz(absres, x=y, dx=0.00001)

    if save:

        cdf_name = 'cdf'
        parity_name = 'cdf_parity'
        dist_name = 'distribution'
        if binsave is not None:
            save = os.path.join(save, 'each_bin')
            cdf_name = '{}_{}'.format(cdf_name, binsave)
            parity_name = '{}_{}'.format(parity_name, binsave)
            dist_name = '{}_{}'.format(dist_name, binsave)

        os.makedirs(save, exist_ok=True)

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

        fig, ax = pl.subplots()

        sns.histplot(
                     z,
                     kde=True,
                     stat='density',
                     color='g',
                     ax=ax,
                     label='Standard Normal Distribution',
                     )

        sns.histplot(
                     x,
                     kde=True,
                     stat='density',
                     color='r',
                     ax=ax,
                     label='Observed Distribution',
                     )

        ax.set_xlabel('z')
        ax.set_ylabel('Fraction')

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
                                 '{}{}.png'.format(dist_name, subsave),
                                 ), bbox_inches='tight')

        fig_legend.savefig(os.path.join(
                                        save,
                                        '{}{}_legend.png'.format(
                                                                 dist_name,
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

    return y, y_pred, areaparity, areacdf, pvalue


def binned_truth(data_cv, metric, bins, gt, save=False):
    '''
    Do analysis on binned data.

    inputs:
        data_cv = Cross validation data.
        meteric = The dissimilarity measure.
        bins = The number of bins to get statistics.
        gt = The ground truth threshold for miscallibration area.
        save = The location to save figures/data.

    outputs:
        out = The statistics from bins.
    '''

    subset = [metric, 'z', 'r/std(y)']
    if metric != 'y_stdc/std(y)':
        subset.append('y_stdc/std(y)')

    out = data_cv[subset].copy()
    out = out.sample(frac=1)

    out['bin'] = pd.qcut(
                         out[metric].rank(method='first'),
                         bins,
                         )

    # Replace bins with integer number in order
    bin_replace = out['bin'].unique()
    bin_replace = sorted(bin_replace)
    bin_replace = enumerate(bin_replace)
    bin_replace = dict(bin_replace)
    bin_replace = {v: k for k, v in bin_replace.items()}
    out['bin'].replace(
                       bin_replace,
                       inplace=True
                       )

    # Bin statistics
    bin_groups = out.groupby('bin', observed=False)
    distmean = bin_groups[metric].mean()
    binmin = bin_groups[metric].min()
    binmax = bin_groups[metric].max()
    zvar = bin_groups['z'].var()
    zmean = bin_groups['z'].mean()
    res = bin_groups['r/std(y)'].mean()
    counts = bin_groups['z'].count()
    rmse = bin_groups['r/std(y)'].apply(lambda x: (sum(x**2)/len(x))**0.5)

    areas = []
    for choice in ['standard_normal']:
        a = bin_groups.apply(lambda x: cdf(
                                           x['z'],
                                           choice=choice,
                                           save=save,
                                           binsave=x['bin'].unique()[0],
                                           )[2:])

        c = a.apply(lambda x: x[2])
        b = a.apply(lambda x: x[1])
        a = a.apply(lambda x: x[0])

        a = a.to_frame().rename({0: choice+'_cdf_parity'}, axis=1)
        b = b.to_frame().rename({0: choice+'_cdf'}, axis=1)
        c = c.to_frame().rename({0: choice+'_pvalue'}, axis=1)

        areas.append(a)
        areas.append(b)
        areas.append(c)

    distmean = distmean.to_frame().add_suffix('_mean')
    binmin = binmin.to_frame().add_suffix('_min')
    binmax = binmax.to_frame().add_suffix('_max')
    zvar = zvar.to_frame().add_suffix('_var')
    zmean = zmean.to_frame().add_suffix('_mean')
    res = res.to_frame().add_suffix('_mean')
    rmse = rmse.to_frame().rename({'r/std(y)': 'rmse/std(y)'}, axis=1)
    counts = counts.to_frame().rename({'z': 'count'}, axis=1)

    out = [
           distmean,
           binmin,
           binmax,
           zvar,
           zmean,
           res,
           rmse,
           counts,
           *areas,
           ]

    if metric != 'y_stdc/std(y)':
        ystdc = bin_groups['y_stdc/std(y)'].mean()
        ystdc = ystdc.to_frame()
        out.append(ystdc)

    out = reduce(lambda x, y: pd.merge(x, y, on='bin'), out)

    out = out.reset_index()
    out.drop('bin', axis=1, inplace=True)

    # Ground truth for bins
    out['id'] = out['standard_normal_cdf'] < gt

    out.sort_values(by=metric+'_mean', inplace=True)

    grouped = [list(group) for key, group in groupby(out['id'])]
    ngroups = len(grouped)

    if save:

        os.makedirs(save, exist_ok=True)

        min_points = out['count'].min()

        in_domain = out['id']
        out_domain = ~in_domain

        pointlabel = 'MPPB = {:.2f}'.format(min_points)
        idpointlabel = 'ID '+pointlabel
        odpointlabel = 'OD '+pointlabel

        zvartot = data_cv['z'].var()
        zmeantot = data_cv['z'].mean()+0.0
        resmeantot = data_cv['r/std(y)'].mean()+0.0

        xchoices = [
                    metric,
                    'standard_normal_cdf',
                    'standard_normal_cdf_parity'
                    ]
        ychoices = [
                    'standard_normal_cdf',
                    'standard_normal_cdf_parity',
                    'z_var',
                    'z_mean',
                    'r/std(y)_mean',
                    'rmse/std(y)',
                    'standard_normal_pvalue',
                    ]

        if metric != 'y_stdc/std(y)':
            xchoices.append('y_stdc/std(y)')
            ychoices.append('y_stdc/std(y)')

        for xchoice in xchoices:

            xmean = xchoice
            if xchoice == metric:
                xmean += '_mean'
                xmin = xchoice+'_min'
                xmax = xchoice+'_max'

            for ychoice in ychoices:

                if ychoice == xchoice:
                    continue

                data = {}
                fig, ax = pl.subplots()

                ax.scatter(
                           out[xmean][in_domain],
                           out[ychoice][in_domain],
                           marker='.',
                           color='g',
                           label=idpointlabel,
                           )

                ax.scatter(
                           out[xmean][out_domain],
                           out[ychoice][out_domain],
                           marker='x',
                           color='r',
                           label=odpointlabel,
                           )

                if xchoice == metric:
                    ax.scatter(
                               out[xmin],
                               out[ychoice],
                               marker='|',
                               color='b',
                               label='Bin Start',
                               )

                    ax.scatter(
                               out[xmax],
                               out[ychoice],
                               marker='|',
                               color='k',
                               label='Bin End',
                               )

                if (
                    ychoice == 'standard_normal_cdf' and
                    xchoice == metric
                   ):

                    ax.axhline(
                               gt,
                               color='r',
                               label='GT = {:.2f}'.format(gt),
                               )
                    data['ground_truth'] = gt

                if ychoice == 'standard_normal_cdf':
                    ax.set_ylabel('Miscalibration Area from CDF')
                    name = 'miscalibration_area_cdf_vs_'

                elif ychoice == 'standard_normal_cdf_parity':
                    ax.set_ylabel('Miscalibration Area from CDF Parity')
                    name = 'miscalibration_area_cdf_parity_vs_'

                elif ychoice == 'same_mean':

                    ax.set_ylabel('Miscalibration Area of Variance')
                    name = 'miscalibration_variance_vs_'

                elif ychoice == 'same_variance':

                    ax.set_ylabel('Miscalibration Area of Mean')
                    name = 'miscalibration_mean_vs_'

                elif ychoice == 'z_var':

                    label = 'Total Variance z = {:.1f}'.format(zvartot)
                    ax.set_ylabel('Variance z')
                    ax.axhline(1.0, color='b', label='Ideal Variance z = 1.0')
                    ax.axhline(
                               zvartot,
                               color='r',
                               label=label
                               )
                    name = 'z_variance_vs_'
                    data['z_var_total'] = zvartot

                elif ychoice == 'z_mean':

                    ax.set_ylabel('Mean z')
                    ax.axhline(0.0, color='b', label='Ideal Mean z = 0.0')
                    ax.axhline(
                               zmeantot,
                               color='r',
                               label='Total Mean z = {:.1f}'.format(zmeantot)
                               )
                    name = 'z_mean_vs_'
                    data['z_mean_total'] = zmeantot

                elif ychoice == 'rmse/std(y)':

                    ax.set_ylabel(r'$RMSE/\sigma_{y}$')
                    name = 'rmse_vs_'

                elif ychoice == 'r/std(y)_mean':
                    ax.set_ylabel(r'Mean $(y-\hat{y})/\sigma_{y}$')
                    label = 'Ideal Mean Residual = 0.0'
                    ax.axhline(0.0, color='b', label=label)
                    label = 'Total Mean Residual = {:.1f}'.format(resmeantot)
                    ax.axhline(
                               resmeantot,
                               color='r',
                               label=label,
                               )
                    name = 'residuals_vs_'
                    data['residual_mean_total'] = resmeantot

                elif ychoice == 'standard_normal_pvalue':
                    ax.set_ylabel('P-Value')
                    name = 'pvalue_vs_'

                if ychoice == 'y_stdc/std(y)':

                    ax.set_ylabel(r'Mean $\sigma_{c}/\sigma_{y}$')
                    name = 'ystdc_vs_'

                if (xchoice == 'y_stdc/std(y)') and (ychoice == 'rmse/std(y)'):

                    x = np.linspace(*ax.get_xlim())
                    ax.plot(x, x, linestyle=':', color='k', label='Ideal')

                if xchoice == 'y_stdc/std(y)':

                    ax.set_xlabel(r'Mean $\sigma_{c}/\sigma_{y}$')
                    name += 'ystdc'

                elif xchoice == 'standard_normal_cdf':

                    ax.set_xlabel('Miscalibration Area from CDF')
                    name += 'miscalibration_area_cdf'

                elif xchoice == 'standard_normal_cdf_parity':

                    ax.set_xlabel('Miscalibration Area from CDF Parity')
                    name += 'miscalibration_area_cdf_parity'

                elif xchoice == 'dist':

                    ax.set_xlabel('Mean D')
                    name += 'dist'

                ax.tick_params(axis='x', rotation=45)

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
                                         '{}.png'.format(name)
                                         ), bbox_inches='tight')
                fig_legend.savefig(os.path.join(
                                                save,
                                                '{}_legend.png'.format(name)
                                                ), bbox_inches='tight')

                pl.close(fig)
                pl.close(fig_legend)

                data['x_id'] = out[xmean][in_domain].tolist()
                data['y_id'] = out[ychoice][in_domain].tolist()
                data['x_od'] = out[xmean][out_domain].tolist()
                data['y_od'] = out[ychoice][out_domain].tolist()

                if metric == xchoice:
                    data['x_min'] = out[xmin].tolist()
                    data['x_max'] = out[xmax].tolist()

                data['mppb'] = int(min_points)

                jsonfile = os.path.join(save, '{}.json'.format(name))
                with open(jsonfile, 'w') as handle:
                    json.dump(data, handle)

        out.to_csv(os.path.join(
                                save,
                                'bin.csv'
                                ), index=False)

    return out


def single_truth(data_cv, metric, save, gt):
    '''
    plot the ground truth with respect to dissimilarity metric.

    inputs:
        data_cv = The cross validation data.
        metric = The dissimilarity metric.
        save = The location to save figures/data.
        gt = The ground truth cutoff.
    '''

    xchoices = [metric]
    ychoices = ['r/std(y)', 'y', 'y_pred']

    if 'z' in data_cv.columns:
        ychoices.append('z')

    df = data_cv[xchoices+ychoices+['id']].copy()
    df['|r/std(y)|'] = df['r/std(y)'].abs()
    ychoices.append('|r/std(y)|')

    in_domain = df['id']
    out_domain = ~in_domain

    os.makedirs(save, exist_ok=True)

    for xchoice in xchoices:
        for ychoice in ychoices:

            data = {}
            fig, ax = pl.subplots()

            ax.scatter(
                       df[xchoice][in_domain],
                       df[ychoice][in_domain],
                       color='g',
                       marker='.',
                       label='ID',
                       )
            ax.scatter(
                       df[xchoice][out_domain],
                       df[ychoice][out_domain],
                       color='r',
                       marker='x',
                       label='OD',
                       )

            if (xchoice == metric) and (ychoice == '|r/std(y)|'):
                ax.axhline(
                           gt,
                           color='r',
                           label='GT = {}'.format(gt),
                           )
                data['gt'] = gt

            if ychoice == '|r/std(y)|':
                ax.set_ylabel(r'$|y-\hat{y}|/\sigma_{y}$')
                name = 'absresidual'
            elif ychoice == 'r/std(y)':
                ax.set_ylabel(r'$(y-\hat{y})/\sigma_{y}$')
                mean = df[xchoice].mean()
                ax.axhline(
                           mean,
                           color='r',
                           label='Mean Residuals = {:.2f}'.format(mean),
                           )
                ax.axhline(
                           0.0,
                           color='b',
                           label='Ideal Mean Residuals = 0.0',
                           )
                data['mean'] = mean
                name = 'residual'
            elif ychoice == 'z':
                mean = df[xchoice].mean()
                ax.axhline(
                           mean,
                           color='r',
                           label='Mean z = {:.2f}'.format(mean),
                           )
                ax.axhline(
                           0.0,
                           color='b',
                           label='Ideal Mean z = 0.0',
                           )
                data['mean'] = mean
                name = 'z'
                ax.set_ylabel('z')

            elif ychoice == 'y':
                name = 'y'
                ax.set_ylabel('y')

            elif ychoice == 'y_pred':
                name = 'ypred'
                ax.set_ylabel(r'$\hat{y}$')

            if xchoice == 'y_stdc/std(y)':
                ax.set_xlabel(r'$\sigma_{c}/\sigma_{y}$')
                name += '_vs_ystdc'
            elif xchoice == 'dist':
                ax.set_xlabel('D')
                name += '_vs_dist'

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

            fig.savefig(
                        os.path.join(save, '{}.png'.format(name)),
                        bbox_inches='tight'
                        )
            fig_legend.savefig(
                               os.path.join(
                                            save,
                                            '{}_legend.png'.format(name)
                                            ),
                               bbox_inches='tight'
                               )
            pl.close(fig)
            pl.close(fig_legend)

            # Repare plot data for saving
            data['x_id'] = df[xchoice][in_domain].tolist()
            data['y_id'] = df[ychoice][in_domain].tolist()
            data['x_od'] = df[xchoice][out_domain].tolist()
            data['y_od'] = df[ychoice][out_domain].tolist()

            jsonfile = os.path.join(save, '{}.json'.format(name))
            with open(jsonfile, 'w') as handle:
                json.dump(data, handle)


def pr(score, in_domain, pos_label, save=False):
    '''
    Plot PR curve and acquire thresholds.

    inputs:
        score = The dissimilarity score.
        in_domain = The label for domain.
        pos_label = The positive label for domain.
        save = The locatin to save the figure/data.

    outputs:
        custom = Data containing threholds for choice of precision/score.
    '''

    if pos_label is True:
        score = -score

    # Sometimes values are not valid because of std(y) = 0.0
    valid_indx = (score != np.inf) & (score != -np.inf)
    score = score[valid_indx]
    in_domain = in_domain[valid_indx]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        prc_scores = precision_recall_curve(
                                            in_domain,
                                            score,
                                            pos_label=pos_label,
                                            )

        precision, recall, thresholds = prc_scores

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        auc_score = average_precision_score(in_domain, score)

    num = 2*recall*precision
    den = recall+precision
    f1_scores = np.divide(
                          num,
                          den,
                          out=np.zeros_like(den), where=(den != 0)
                          )

    # Maximum F1 score
    max_f1_index = np.argmax(f1_scores)

    custom = {}
    custom['Max F1'] = {
                        'Precision': precision[max_f1_index],
                        'Recall': recall[max_f1_index],
                        'Threshold': thresholds[max_f1_index],
                        'F1': f1_scores[max_f1_index],
                        }

    # Loop for lowest to highest to get better thresholds than the other way
    nprec = len(precision)
    nthresh = nprec-1  # sklearn convention
    nthreshindex = nthresh-1  # Foor loop index comparison
    loop = range(nprec)
    for cut in [0.95]:

        # Correction for no observed precision higher than cut
        if not any(precision[:-1] >= cut):
            break
        else:
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

        # If precision is set at arbitrary 1 from sklearn convention
        if index > nthreshindex:
            custom[name]['Threshold'] = max(thresholds)
        else:
            custom[name]['Threshold'] = thresholds[index]

    # Convert back
    if pos_label is True:
        thresholds = -thresholds
        score = -score
        for key, value in custom.items():
            custom[key]['Threshold'] *= -1

    if save:

        baseline = [1 if i == pos_label else 0 for i in in_domain]
        baseline = sum(baseline)/len(in_domain)
        relative_base = 1-baseline  # The amount of area to gain in PR
        diff = auc_score-baseline

        # AUC relative to the baseline
        if relative_base == 0.0:
            auc_relative = 0.0
        else:
            auc_relative = (diff)/relative_base

        os.makedirs(save, exist_ok=True)

        fig, ax = pl.subplots()

        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_label = 'AUC: {:.2f}\n'.format(auc_score)
        pr_label += 'Relative AUC: {:.2f}\n'.format(auc_relative)
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

        for key, values in custom.items():
            p = custom[key]['Precision']
            r = custom[key]['Recall']
            t = custom[key]['Threshold']

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
        fig.savefig(os.path.join(save, 'pr.png'), bbox_inches='tight')
        fig_legend.savefig(os.path.join(
                                        save,
                                        'pr_legend.png'
                                        ), bbox_inches='tight'
                           )
        pl.close(fig)
        pl.close(fig_legend)

        # Repare plot data for saving
        data = {}
        data['recall'] = list(recall)
        data['precision'] = list(precision)
        data['baseline'] = baseline
        data['auc'] = auc_score
        data['auc-baseline'] = diff
        data['auc_relative'] = auc_relative
        data.update(custom)

        jsonfile = os.path.join(save, 'pr.json')
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
