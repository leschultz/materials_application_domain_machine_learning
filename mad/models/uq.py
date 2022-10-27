from scipy.optimize import minimize
import numpy as np


def llh(std, res, x, func):
    '''
    Compute the log likelihood.
    '''

    total = np.log(2*np.pi)
    total += np.log(func(x, std)**2)
    total += (res**2)/(func(x, std)**2)
    total *= -0.5

    return total


def set_llh(y, y_pred, y_std, x, func):
    '''
    Compute the log likelihood for a dataset.
    '''

    res = y-y_pred

    # Get negative to use minimization instead of maximization of llh
    opt = minimize(
                   lambda x: -sum(llh(y_std, res, x, func))/len(res),
                   x,
                   method='nelder-mead',
                   )

    params = opt.x

    return params


# Polynomial given coefficients
def poly(c, std):
    total = 0.0
    for i in range(len(c)):
        total += c[i]*std**i
    return abs(total)


# Power function
def power(c, std):
    return abs(c[0]*std**c[1])


class ensemble_model:

    def __init__(self, params=[0.0, 1.0], uq_func=poly):
        self.params = params
        self.uq_func = uq_func

    def fit(self, y, y_pred, y_std):

        params = set_llh(
                         y,
                         y_pred,
                         y_std,
                         self.params,
                         self.uq_func
                         )

        self.params = params

    def predict(self, y_std):
        return self.uq_func(self.params, y_std)
