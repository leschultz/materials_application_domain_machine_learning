from sklearn.metrics import (
                             precision_recall_curve,
                             average_precision_score,
                             mean_squared_error,
                             )

from scipy.optimize import minimize
from functools import reduce

import pandas as pd
import numpy as np
import warnings
import copy

# Standard normal distribution
nz = 10000
z_standard_normal = np.random.normal(0, 1, nz)


def cdf(x):
    '''
    Plot the quantile quantile plot for cummulative distributions.

    inputs:
        x = The residuals normalized by the calibrated uncertainties.

    outputs:
        y = The cummulative distribution of observed data.
        y_pred = The cummulative distribution of standard normal distribution.
        area = The area between y and y_pred.
    '''

    nx = len(x)

    # Need sorting
    x = sorted(x)
    z = sorted(z_standard_normal)

    # Cummulative fractions
    xfrac = np.arange(1, nx+1)/(nx)
    zfrac = np.arange(1, nz+1)/(nz)

    # Interpolation to compare cdf
    eval_points = sorted(list(set(x+z)))
    y_pred = np.interp(eval_points, x, xfrac)  # Predicted
    y = np.interp(eval_points, z, zfrac)  # Standard Normal

    # Area between ideal distribution and observed
    absres = np.abs(y_pred-y)
    areacdf = np.trapz(absres, x=eval_points, dx=0.00001)

    return eval_points, y, y_pred, areacdf


def llh(std, res, x, func):
    '''
    Compute the log likelihood.

    inputs:
        std = The uncertainty measure.
        res = The residuals.
        x = The initial fitting parameters.
        func = The type of UQ function.

    outputs:
        total =  The total negative log likelihood.
    '''

    total = np.log(2*np.pi)
    total += np.log(func(x, std)**2)
    total += (res**2)/(func(x, std)**2)
    total *= -0.5

    return total


def set_llh(y, y_pred, y_std, x, func):
    '''
    Compute the log likelihood for a dataset.

    inputs:
        y = The target variable.
        y_pred = The prediction of the target variable.
        y_std = The uncertainty of the target variable.
        x = The initial guess for UQ fitting parameters.
        func = The UQ fitting function.

    outputs
        params = The fit parameters for the UQ function.
    '''

    res = y-y_pred

    # Get negative to use minimization instead of maximization of llh
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        opt = minimize(
                       lambda x: -sum(llh(y_std, res, x, func))/len(res),
                       x,
                       method='nelder-mead',
                       )

    params = opt.x

    return params


def poly(c, std):
    '''
    A polynomial function.

    inputs:
        c = A list for each polynomial coefficient.
        std = The UQ measure.

    outputs:
        total = The aggregate absolute value result from the polynomial.
    '''

    total = 0.0
    for i in range(len(c)):
        total += c[i]*std**i
    total = abs(total)

    return total


def pr(d, labels, precs):
    '''
    Precision recall curve.
    '''

    # Compensate for all classes being ID/OD
    if all(i == 'ID' for i in labels):
        precision = np.array([1.0, 1.0])
        recall = np.array([0.0, 1.0])
        thresholds = np.repeat(np.inf, 2)
        auc_score = 1.0

    elif all(i == 'OD' for i in labels):
        precision = np.array([0.0, 0.0])
        recall = np.array([0.0, 1.0])
        thresholds = np.array([-np.inf])
        auc_score = 0.0

    else:
        d = -d  # Because lowest d is more likely ID
        prc_scores = precision_recall_curve(
                                            labels,
                                            d,
                                            pos_label='ID',
                                            )

        precision, recall, thresholds = prc_scores

        auc_score = average_precision_score(
                                            labels,
                                            d,
                                            pos_label='ID',
                                            )+0.0  # Make positive

        thresholds *= -1

    num = 2*recall*precision
    den = recall+precision
    f1_scores = np.divide(
                          num,
                          den,
                          out=np.zeros_like(den), where=(den != 0)
                          )
    # Save data
    data = {}

    # Maximum F1 score
    max_f1_index = np.argmax(f1_scores)

    data['Max F1'] = {
                      'Precision': precision[max_f1_index],
                      'Recall': recall[max_f1_index],
                      'Threshold': thresholds[max_f1_index],
                      'F1': f1_scores[max_f1_index],
                      }

    # Loop for lowest to highest to get better thresholds
    nprec = len(precision)
    nthresh = nprec-1  # sklearn convention
    nthreshindex = nthresh-1  # Foor loop index comparison
    loop = range(nprec)
    for cut in precs:

        for index in loop:
            p = precision[index]
            if p >= cut:
                break

        name = 'Minimum Precision: {}'.format(cut)
        data[name] = {
                      'Precision': precision[index],
                      'Recall': recall[index],
                      'F1': f1_scores[index],
                      }

        # If precision is set at arbitrary 1 from sklearn convention
        if index > nthreshindex:
            data[name]['Threshold'] = max(thresholds)
        else:
            data[name]['Threshold'] = thresholds[index]

    data['AUC'] = auc_score
    data['Baseline'] = np.sum(labels == 'ID')/labels.shape[0]
    data['AUC-Baseline'] = (auc_score-data['Baseline'])+0.0  # Force positive
    data['Precision'] = precision.tolist()
    data['Recall'] = recall.tolist()
    data['Thresholds'] = thresholds.tolist()

    return data


def bin_data(data_cv, bins, by='d_pred'):

    # Copy to prevent problems
    data_cv = copy.deepcopy(data_cv)
    data_cv = data_cv.sample(frac=1)  # Shuffle to prevent sorting bias

    # Correct for cases were many cases are at the same value
    count = 0
    unique_quantiles = 0
    max_iters = 100
    while unique_quantiles < bins:
        quantiles = pd.qcut(
                            data_cv[by],
                            bins+count,
                            duplicates='drop',
                            )
        unique_quantiles = len(quantiles.unique())

        if count >= max_iters:
            break

        count += 1

    data_cv['bin'] = quantiles

    # Calculate statistics
    bin_groups = data_cv.groupby('bin', observed=False, sort=False)
    distmean = bin_groups['d_pred'].mean()
    binmax = bin_groups['d_pred'].max()
    counts = bin_groups['z'].count()
    stdc = bin_groups['y_stdc_pred/std_y'].mean()
    rmse = bin_groups['r/std_y'].apply(lambda x: (sum(x**2)/len(x))**0.5)

    area = bin_groups.apply(lambda x: cdf(
                                          x['z'],
                                          )[-1])

    area = area.to_frame().rename({0: 'cdf_area'}, axis=1)

    distmean = distmean.to_frame().add_suffix('_mean')
    binmax = binmax.to_frame().add_suffix('_max')
    stdc = stdc.to_frame().add_suffix('_mean')
    rmse = rmse.to_frame().rename({'r/std_y': 'rmse/std_y'}, axis=1)
    counts = counts.to_frame().rename({'z': 'count'}, axis=1)

    # Combine data for each bin
    bin_cv = [
              counts,
              distmean,
              binmax,
              stdc,
              rmse,
              area,
              ]

    bin_cv = reduce(lambda x, y: pd.merge(x, y, on='bin'), bin_cv)

    bin_cv = bin_cv.reset_index()

    return data_cv, bin_cv


def ground_truth(self, y):

    # Acquire ground truths
    if any([self.gt_rmse is None, self.gt_area is None]):
        std_y = np.std(y)
        mean_y = np.mean(y)

    if self.gt_rmse is None:
        mean = np.repeat(mean_y, y.shape[0])
        naive_rmse = mean_squared_error(
                                        y,
                                        mean,
                                        )
        naive_rmse **= 0.5
        naive_rmse /= std_y

        self.gt_rmse = naive_rmse

    if self.gt_area is None:
        naive_area = y-mean_y
        naive_area /= std_y
        naive_area = cdf(naive_area)
        naive_area = naive_area[3]

        self.gt_area = naive_area

    return self
