from sklearn.preprocessing import StandardScaler
from mad.functions import parallel, find
from matplotlib import pyplot as pl
import statsmodels.api as sm
import pandas as pd
import numpy as np

import matplotlib
import os

# Font styles
font = {'font.size': 16, 'lines.markersize': 10}
matplotlib.rcParams.update(font)


def cdf_parity(x, ax, fig, color):
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

    y_pred = y_pred.tolist()
    y = y.tolist()

    ax.plot(
            y,
            y_pred,
            zorder=0,
            color=color,
            label='Area: {:.3f}'.format(area)
            )

    # Line of best fit
    ax.plot(
            [0, 1],
            [0, 1],
            color='k',
            linestyle=':',
            zorder=1,
            )

    ax.legend()
    ax.set_ylabel('Predicted CDF')
    ax.set_xlabel('Standard Normal CDF')

    h = 8
    w = 8

    fig.set_size_inches(h, w, forward=True)
    ax.set_aspect('equal')


def main():

    percentile=1

    df = pd.concat(parallel(pd.read_csv, find('.', 'assessment.csv')))
    #df['gpr_std'] = df['coral']

    fig_train_ground, ax_train_ground = pl.subplots()
    fig_train_assess, ax_train_assess = pl.subplots()
    fig_train_cal, ax_train_cal = pl.subplots()
    fig_test_ground, ax_test_ground = pl.subplots()
    fig_test_assess, ax_test_assess = pl.subplots()
    fig_test_cal, ax_test_cal = pl.subplots()
    for group, values in df.groupby(['fold']):
        print(values)

        group = [group]
        name = list(map(str, group))
        name = '_'.join(name)
        print(name)

        os.makedirs('figures', exist_ok=True)

        name = os.path.join('figures', name)

        train = values[values['split'] == 'cv']
        test = values[values['split'] == 'test']

        stdy = train['y'].std()
        y = train['y'].values/stdy
        y_pred = train['y_pred'].values/stdy
        gpr_std = train['dist'].values
        stdcal = train['y_std'].values/stdy

        gpr_std = -np.log10(1e-8+1-gpr_std)

        res = y-y_pred
        absres = abs(res)
        err = abs(absres-stdcal)

        scaler = StandardScaler()
        vals = np.array([stdcal, absres]).T
        scaler.fit(vals)
        vals = scaler.transform(vals)
        kde = sm.nonparametric.KDEMultivariate(vals, var_type='cc')
        pdf = kde.pdf(vals)

        cut = np.percentile(pdf, percentile)
        in_domain = (pdf > cut)
        out_domain = ~in_domain

        fig, ax = pl.subplots()
        ax.scatter(gpr_std[in_domain], stdcal[in_domain], color='g')
        ax.scatter(gpr_std[out_domain], stdcal[out_domain], color='r')
        ax_train_assess.scatter(gpr_std[in_domain], stdcal[in_domain], color='g', zorder=1)
        ax_train_assess.scatter(gpr_std[out_domain], stdcal[out_domain], color='r', zorder=2)
        ax.set_ylabel('$\sigma_{c}/\sigma_{y}$')
        ax.set_xlabel('$-log_{10}(1e-8+1-GPR_{\sigma})$')
        ax.set_xlim([0, 8.1])
        ax.set_title('Train/CV')
        fig.savefig(name+'_train_assessment')
        pl.close(fig)

        fig, ax = pl.subplots()
        ax.scatter(absres[in_domain], stdcal[in_domain], color='g')
        ax.scatter(absres[out_domain], stdcal[out_domain], color='r')
        ax_train_ground.scatter(absres[in_domain], stdcal[in_domain], color='g', zorder=1)
        ax_train_ground.scatter(absres[out_domain], stdcal[out_domain], color='r', zorder=2)
        ax.set_xlabel('$|y-\hat{y}|/\sigma_{y}$')
        ax.set_ylabel('$\sigma_{c}/\sigma_{y}$')
        ax.set_title('Train/CV')
        fig.savefig(name+'_train_ground_truth')
        pl.close(fig)

        fig, ax = pl.subplots()
        x = res/stdcal
        cdf_parity(x[in_domain], ax, fig, color='g')
        cdf_parity(x[out_domain], ax, fig, color='r')
        cdf_parity(x, ax, fig, color='b')
        cdf_parity(x[in_domain], ax_train_cal, fig_train_cal, color='g')
        cdf_parity(x[out_domain], ax_train_cal, fig_train_cal, color='r')
        cdf_parity(x, ax_train_cal, fig_train_cal, color='b')
        ax.set_title('Train/CV')
        fig.savefig(name+'_train_calibration')
        pl.close(fig)

        y = test['y'].values/stdy
        y_pred = test['y_pred'].values/stdy
        gpr_std = test['dist'].values
        stdcal = test['y_std'].values/stdy

        gpr_std = -np.log10(1e-8+1-gpr_std)

        res = y-y_pred
        absres = abs(res)
        err = abs(absres-stdcal)

        vals = np.array([stdcal, absres]).T
        vals = scaler.transform(vals)
        pdf = kde.pdf(vals)

        in_domain = (pdf > cut)
        out_domain = ~in_domain

        fig, ax = pl.subplots()
        ax.scatter(gpr_std[in_domain], stdcal[in_domain], color='g')
        ax.scatter(gpr_std[out_domain], stdcal[out_domain], color='r')
        ax_test_assess.scatter(gpr_std[in_domain], stdcal[in_domain], color='g', zorder=1)
        ax_test_assess.scatter(gpr_std[out_domain], stdcal[out_domain], color='r', zorder=2)
        ax.set_ylabel('$\sigma_{c}/\sigma_{y}$')
        ax.set_xlabel('$-log_{10}(1e-8+1-GPR_{\sigma})$')
        ax.set_xlim([0, 8.1])
        ax.set_title('Test')
        fig.savefig(name+'_test_assessment')
        pl.close(fig)

        fig, ax = pl.subplots()
        ax.scatter(absres[in_domain], stdcal[in_domain], color='g')
        ax.scatter(absres[out_domain], stdcal[out_domain], color='r')
        ax_test_ground.scatter(absres[in_domain], stdcal[in_domain], color='g', zorder=1)
        ax_test_ground.scatter(absres[out_domain], stdcal[out_domain], color='r', zorder=2)
        ax.set_xlabel('$|y-\hat{y}|/\sigma_{y}$')
        ax.set_ylabel('$\sigma_{c}/\sigma_{y}$')
        ax.set_title('Test')
        fig.savefig(name+'_test_ground_truth')
        pl.close(fig)

        fig, ax = pl.subplots()
        x = res/stdcal
        cdf_parity(x[in_domain], ax, fig, color='g')
        cdf_parity(x[in_domain], ax_test_cal, fig_test_cal, color='g')

        if sum(out_domain) > 0:
            cdf_parity(x[out_domain], ax, fig, color='r')
            cdf_parity(x[out_domain], ax_test_cal, fig_test_cal, color='r')

        cdf_parity(x, ax, fig, color='b')
        cdf_parity(x, ax_test_cal, fig_test_cal, color='b')
        ax.set_title('Test')
        fig.savefig(name+'_test_calibration')
        pl.close(fig)

    ax_train_assess.set_ylabel('$\sigma_{c}/\sigma_{y}$')
    ax_train_assess.set_xlabel('$GPR(\sigma)$')
    ax_train_assess.set_xlim([0, 8.1])
    ax_train_assess.set_title('Train/CV')
    fig_train_assess.savefig('figures/total_train_assessment')
    
    ax_train_ground.set_xlabel('$|y-\hat{y}|/\sigma_{y}$')
    ax_train_ground.set_ylabel('$\sigma_{c}/\sigma_{y}$')
    ax_train_ground.set_title('Train/CV')
    fig_train_ground.savefig('figures/total_train_ground_truth')

    ax_train_cal.set_title('Train/CV')
    fig_train_cal.savefig('figures/total_train_calibration')

    ax_test_assess.set_ylabel('$\sigma_{c}/\sigma_{y}$')
    ax_test_assess.set_xlabel('$GPR(\sigma)$')
    ax_test_assess.set_xlim([0, 8.1])
    ax_test_assess.set_title('Test')
    fig_test_assess.savefig('figures/total_test_assessment')

    ax_test_ground.set_xlabel('$|y-\hat{y}|/\sigma_{y}$')
    ax_test_ground.set_ylabel('$\sigma_{c}/\sigma_{y}$')
    ax_test_ground.set_title('Test')
    fig_test_ground.savefig('figures/total_test_ground_truth')

    ax_test_cal.set_title('Test')
    fig_test_cal.savefig('figures/total_test_calibration')


if __name__ == '__main__':
    main()
