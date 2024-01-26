from sklearn.metrics import (
                             precision_recall_curve,
                             average_precision_score,
                             )

from sklearn.model_selection import RepeatedKFold
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from madml.utils import parallel
from sklearn.base import clone
from functools import reduce
from sklearn import metrics
from scipy import stats

import pandas as pd
import numpy as np
import warnings
import copy

# Standard normal distribution
nz = 10000
z_standard_normal = np.random.normal(0, 1, nz)
pd.options.mode.chained_assignment = None


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


class calibration:
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

        if (self.params is not None) and (self.prior is None):
            self.params = set_llh(
                                  y,
                                  y_pred,
                                  y_std,
                                  self.params,
                                  self.uq_func
                                  )
        elif self.prior is None:
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


class dissimilarity:

    def __init__(
                 self,
                 dis='kde',
                 kernel='epanechnikov',
                 bandwidth=None,
                 ):

        self.dis = dis
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(
            self,
            X_train,
            ):
        '''
        Get the dissimilarities based on a metric.

        inputs:
            X_train = The features of the training set.
            dis = The distance to consider.
            model = The model to get feature importances.
            n_features = The number of features to keep.

        ouputs:
            d = Dissimilarities.
        '''

        if self.dis == 'kde':

            if self.bandwidth is None:
                self.bandwidth = estimate_bandwidth(X_train)

            if self.bandwidth > 0.0:
                model = KernelDensity(
                                      kernel=self.kernel,
                                      bandwidth=self.bandwidth,
                                      )

                model.fit(X_train)

                dis = model.score_samples(X_train)
                m = np.max(dis)

                def pred(X):
                    out = model.score_samples(X)
                    out = out-m
                    out = np.exp(out)
                    out = 1-out
                    out = np.maximum(0.0, out)
                    return out

                self.model = pred

            else:
                self.model = lambda x: np.repeat(1.0, len(x))

        else:

            def pred(X):
                out = cdist(X_train, X, self.dis)
                out = np.mean(out, axis=0)
                return out

            self.model = pred

    def predict(self, X):
        '''
        Get the dissimilarities for individual cases.

        inputs:
            X = The features of data.
        outputs:
            d = The Dissimilarites.
        '''

        d = self.model(X)

        return d


class domain:
    '''
    Domain classification model.
    '''

    def __init__(self, precs=[0.95]):

        self.precs = precs

    def train(self, d, labels):
        '''
        Train the domain model on dissimilarity scores by finding thresholds.
        '''

        d = -d  # Because lowest d is more likely ID

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            prc_scores = precision_recall_curve(
                                                labels,
                                                d,
                                                pos_label=True,
                                                )

            precision, recall, thresholds = prc_scores

            auc_score = average_precision_score(
                                                labels,
                                                d,
                                                pos_label=True,
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

        data = {}
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
        for cut in self.precs:

            # Correction for no observed precision higher than cut
            if not any(precision[:-1] >= cut):
                break
            else:
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

        # Convert back
        for key, value in data.items():
            data[key]['Threshold'] *= -1

        self.data = data

    def predict(self, d):
        '''
        Predict the domain based on thresholds.

        inputs:
            d = The score.

        outputs:
            do_pred = The domain prediction.
        '''

        do_pred = pd.DataFrame()
        for key, value in self.data.items():

            key = 'Domain Prediction from {}'.format(key)
            cut = value['Threshold']
            do_pred[key] = d <= cut

        return do_pred


def pipe_transforms(model, X):
    '''
    Apply all steps from a pipeline before the final prediction.

    inputs:
        model = The gridsearch pipeline model.
        X = The features.
    ouputs:
        X = The features with applied transformations.
    '''

    for step in list(model.best_estimator_.named_steps)[:-1]:

        step = model.best_estimator_.named_steps[step]
        X = step.transform(X)

    return X


def predict_std(model, X):
    '''
    Get predictions from ensemble model.

    inputs:
        model = The gridsearch pipeline model.
        X = The featues.

    outputs:
        std = The standard deviation between models from ensemble model.
    '''

    estimators = model.best_estimator_
    estimators = estimators.named_steps['model']
    estimators = estimators.estimators_

    std = []
    for i in estimators:
        std.append(i.predict(X))

    std = np.std(std, axis=0)

    return std


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
    areaparity = np.trapz(absres, x=y, dx=0.00001)

    return y, y_pred, areaparity, areacdf


def bin_data(data_cv, bins, by):

    # Correct for cases were many cases are at the same value
    indx = data_cv[by] < 1.0

    # Bin data by our dissimilarity
    data_cv.loc[:, 'bin'] = '[1.0, 1.0]'
    sub_bin = pd.qcut(
                      data_cv[indx][by],
                      bins-1,
                      )

    data_cv.loc[indx, 'bin'] = sub_bin

    # Calculate statistics
    bin_groups = data_cv.groupby('bin', observed=False)
    distmean = bin_groups[by].mean()
    binmin = bin_groups[by].min()
    binmax = bin_groups[by].max()
    counts = bin_groups['z'].count()
    rmse = bin_groups['r/std_y'].apply(lambda x: (sum(x**2)/len(x))**0.5)

    area = bin_groups.apply(lambda x: cdf(
                                          x['z'],
                                          )[-1])

    area = area.to_frame().rename({0: 'cdf_area'}, axis=1)

    distmean = distmean.to_frame().add_suffix('_mean')
    binmin = binmin.to_frame().add_suffix('_min')
    binmax = binmax.to_frame().add_suffix('_max')
    rmse = rmse.to_frame().rename({'r/std_y': 'rmse/std_y'}, axis=1)
    counts = counts.to_frame().rename({'z': 'count'}, axis=1)

    # Combine data for each bin
    bin_cv = [
              distmean,
              binmin,
              binmax,
              counts,
              rmse,
              area,
              ]

    bin_cv = reduce(lambda x, y: pd.merge(x, y, on='bin'), bin_cv)

    bin_cv = bin_cv.reset_index()

    return bin_cv


def assign_ground_truth(data_cv, bin_cv, gt_rmse, gt_area):

    bin_cv['domain_rmse/sigma_y'] = bin_cv['rmse/std_y'] <= gt_rmse
    bin_cv['domain_cdf_area'] = bin_cv['cdf_area'] <= gt_area

    cols = [
            'domain_rmse/sigma_y',
            'domain_cdf_area',
            ]

    # Allocate data
    for col in cols:
        data_cv[col] = None

    # Assign bin data to individual points
    for i in bin_cv.bin:

        # Ground labels based on rmse
        row = data_cv['bin'] == i
        gt = bin_cv.loc[bin_cv['bin'] == i][cols]

        for col in cols:
            data_cv.loc[row, col] = gt[col].values[0]


class combine:

    '''
    Combine distance, UQ, and ensemble regression models.
    '''

    def __init__(
                 self,
                 gs_model,
                 ds_model,
                 uq_model,
                 splits=[('fit', RepeatedKFold(n_repeats=2))],
                 bins=10,
                 precs=[0.95],
                 disable_tqdm=False,
                 ):

        '''
        inputs:
            gs_model = The grid search enemble model.
            ds_model = The distance model.
            uq_model = The UQ model
            splits = The list of splitting generators.
            bins = The number of quantailes for binning data.
        '''

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model
        self.bins = bins
        self.splits = copy.deepcopy(splits)
        self.precs = precs
        self.disable_tqdm = disable_tqdm

        # Add a splitter to calibrate UQ and prevent overfitting
        uqsplits = []
        for i in self.splits:
            if 'fit' == i[0]:
                i = copy.deepcopy(i[1])
                i = ('calibration', i)
                uqsplits.append(i)

        self.splits += uqsplits

    def cv(self, split, gs_model, ds_model, X, y, g=None):
        '''
        Do cross validation.

        inputs:
            split = The split of data.
            gs_model = The grisearch pipeline model.
            ds_model = The distance model.
            X = The features.
            y = The target variable.
            g = The groups for data.
            split = The split of data.

        outputs:
            data = A pandas dataframe containing CV results.
        '''

        count, splitter, tr, te = split

        if (tr.shape[0] < 1) | (te.shape[0] < 1):
            return pd.DataFrame()

        # Copy/clone models to prefent leakage
        ds_model_cv = copy.deepcopy(ds_model)
        gs_model_cv = clone(gs_model)

        gs_model_cv.fit(X[tr], y[tr])

        # Get scaled features
        X_trans_tr = pipe_transforms(gs_model_cv, X[tr])
        X_trans_te = pipe_transforms(gs_model_cv, X[te])

        ds_model_cv.fit(X_trans_tr)

        # Variable to save data
        data = pd.DataFrame()

        # Starting values
        data['index'] = te.astype(int)
        data['splitter'] = splitter
        data['fold'] = count
        data['y'] = y[te]

        # Statistics from training data
        data['std_y'] = np.std(y[tr])

        # Predictions
        data['y_pred'] = gs_model_cv.predict(X[te])
        data['y_stdu'] = predict_std(gs_model_cv, X_trans_te)
        data['d'] = ds_model_cv.predict(X_trans_te)
        data['r'] = y[te]-data['y_pred']
        data['r/std_y'] = data['r']/data['std_y']

        return data

    def fit(self, X, y, g=None):
        '''
        Fit all models. Thresholds for domain classification are also set.

        inputs:
            X = The features.
            y = The target variable.
            g = The groups.

        outputs:
            data_cv = Cross validation data used.
            data_cv_bin = The binned cross validation data.
        '''

        # Generate all splits
        splits = []
        for i in self.splits:
            for count, (tr, te) in enumerate(i[1].split(X, y, g)):
                splits.append((count, i[0], tr, te))

        # Analyze each split in parallel
        data_cv = parallel(
                           self.cv,
                           splits,
                           disable=self.disable_tqdm,
                           gs_model=self.gs_model,
                           ds_model=self.ds_model,
                           X=X,
                           y=y,
                           g=g,
                           )

        # Combine data
        data_cv = pd.concat(data_cv)

        # Separate data
        data_id = data_cv[data_cv['splitter'] == 'calibration']

        # Fit models
        self.gs_model.fit(X, y)

        self.uq_model.fit(
                          data_id['y'].values,
                          data_id['y_pred'].values,
                          data_id['y_stdu'].values
                          )

        X_trans = pipe_transforms(
                                  self.gs_model,
                                  X,
                                  )

        self.ds_model.fit(X_trans)

        # Update data not used for calibration of UQ
        data_cv['y_stdc'] = self.uq_model.predict(data_cv['y_stdu'].values)
        data_cv['z'] = data_cv['r']/data_cv['y_stdc']

        # Separate out of bag data from those used to fint UQ
        data_id = data_cv[data_cv['splitter'] == 'calibration']
        data_cv = data_cv[data_cv['splitter'] != 'calibration']

        # Get binned data from alternate forms of sampling
        bin_id = bin_data(data_id, self.bins, 'd')
        bin_cv = bin_data(data_cv, self.bins, 'd')

        # Acquire ground truths
        self.gt_rmse = bin_id['rmse/std_y'].max()
        self.gt_area = bin_id[bin_id['bin'] != '[1.0, 1.0]']['cdf_area'].max()

        # Classify ground truth labels
        assign_ground_truth(data_cv, bin_cv, self.gt_rmse, self.gt_area)

        # Fit domain classifiers
        self.domain_rmse = domain(self.precs)
        self.domain_area = domain(self.precs)

        # Train classifiers
        self.domain_rmse.train(bin_cv['d_max'], bin_cv['domain_rmse/sigma_y'])
        self.domain_area.train(bin_cv['d_max'], bin_cv['domain_cdf_area'])

        self.data_cv = data_cv
        self.bin_cv = bin_cv

    def combine_domains_preds(self, d):
        '''
        Combine domain classifiers that were fit for RMSE
        and miscalibration area.
        '''

        # Predict domains on training data
        data_rmse_dom_pred = self.domain_rmse.predict(d)
        data_rmse_dom_pred = data_rmse_dom_pred.add_prefix('rmse/sigma_y ')

        data_area_dom_pred = self.domain_area.predict(d)
        data_area_dom_pred = data_area_dom_pred.add_prefix('cdf_area ')

        dom_pred = pd.concat([
                              data_rmse_dom_pred,
                              data_area_dom_pred,
                              ], axis=1)

        return dom_pred

    def predict(self, X):
        '''
        Aggregate all predictions from models.
        '''

        # Transform data
        X_trans = pipe_transforms(self.gs_model, X)

        # Predict from each model
        pred = pd.DataFrame()
        pred['y_pred'] = self.gs_model.predict(X)
        pred['d_pred'] = self.ds_model.predict(X_trans)
        pred['y_stdu_pred'] = predict_std(self.gs_model, X_trans)
        pred['y_stdc_pred'] = self.uq_model.predict(pred['y_stdu_pred'])

        pred = pd.concat([
                          pred,
                          self.combine_domains_preds(pred['d_pred']),
                          ], axis=1)

        return pred