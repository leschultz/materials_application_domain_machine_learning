from scipy.optimize import minimize
import numpy as np
import warnings


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


def power(c, std):
    '''
    A function raised to a power and multiplied by a number.

    inputs:
        c = The multiplying coefficient.
        std = The power to raise the input.

    outputs:
        total = The multiple power of the input.
    '''

    total = abs(c[0]*std**c[1])

    return total


class calibration_model:
    '''
    A UQ model for calibration of uncertainties.
    '''

    def __init__(
                 self,
                 uq_func=poly,
                 params=None,
                 prior=None,
                 ):
        '''
        inputs:
            uq_func = The type of UQ function.
            params = The fitting coefficients initial guess.
            prior = A prior function fit to predict values.
        '''

        self.params = params
        self.uq_func = uq_func
        self.prior = prior

    def fit(self, y, y_pred, y_std):
        '''
        Fit the UQ parameters for a model.

        inputs:
            y = The target variable.
            y_pred = The prediction for the target variable.
            y_std = The uncertainty for the target variable.
        '''

        if self.params is not None:
            self.params = set_llh(
                                  y,
                                  y_pred,
                                  y_std,
                                  self.params,
                                  self.uq_func
                                  )
        elif self.prior is not None:
            self.params = 'Manual'
        else:
            self.uq_func.fit(y_std.reshape(-1, 1), abs(y-y_pred))

    def predict(self, y_std):
        '''
        Use the fitted UQ model to predict uncertainties.

        inputs:
            y_std = The uncalibrated uncertainties.

        outputs:
            y_stdc = The calibrated uncertainties.
        '''

        if self.params is not None:
            y_stdc = self.uq_func(self.params, y_std)
        elif self.prior is not None:
            y_stdc = self.uq_func(y_std)
        else:
            y_stdc = self.uq_func.predict(y_std.reshape(-1, 1))

        return y_stdc
