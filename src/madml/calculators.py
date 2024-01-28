from scipy.optimize import minimize

import pandas as pd
import numpy as np
import warnings

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
