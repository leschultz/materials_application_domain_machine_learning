from madml.calculators import (
                               ground_truth,
                               bin_data,
                               set_llh,
                               poly,
                               pr,
                               )

from sklearn.model_selection import RepeatedKFold
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from madml.utils import parallel
from madml.plots import plotter
from sklearn.base import clone

import pandas as pd
import numpy as np
import copy

pd.options.mode.chained_assignment = None


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

    def __init__(self, precs=[]):

        self.precs = precs

    def fit(self, d, labels):
        '''
        Train the domain model on dissimilarity scores by finding thresholds.
        '''

        self.data = pr(d, labels, self.precs)

    def predict(self, d, d_input=None):
        '''
        Predict the domain based on thresholds.

        inputs:
            d = The score.
            d_input = A user defined cutoff on d.

        outputs:
            do_pred = The domain prediction.
        '''

        # Keys that are not prediction thresholds
        skip = [
                'Precision',
                'Recall',
                'Thresholds',
                'AUC',
                'Baseline',
                'AUC-Baseline',
                ]

        do_pred = pd.DataFrame()
        for key, value in self.data.items():

            if key in skip:
                continue

            key = 'Domain Prediction from {}'.format(key)
            cut = value['Threshold']
            do_pred[key] = np.where(d <= cut, 'ID', 'OD')

        if d_input is not None:
            do_pred['d_input'] = np.where(d <= d_input, 'ID', 'OD')

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


def assign_ground_truth(data_cv, bin_cv):

    data_cv = copy.deepcopy(data_cv)
    bin_cv = copy.deepcopy(bin_cv)

    data_cv = data_cv.merge(bin_cv, on=['bin'])

    # Innitiate arrays
    cols = ['gt_rmse', 'gt_area']
    for c in cols:
        bin_cv[c] = None

    # Propagate ground truths
    for group, value in data_cv.groupby(['bin', *cols], observed=True):
        row = bin_cv['bin'] == group[0]
        bin_cv.loc[row, 'gt_rmse'] = group[1]
        bin_cv.loc[row, 'gt_area'] = group[2]

    # Make labels
    rmse = data_cv['rmse/std_y'] <= data_cv['gt_rmse']
    area = data_cv['cdf_area'] <= data_cv['gt_area']

    data_cv['domain_rmse/std_y'] = np.where(rmse, 'ID', 'OD')
    data_cv['domain_cdf_area'] = np.where(area, 'ID', 'OD')

    rmse = bin_cv['rmse/std_y'] <= bin_cv['gt_rmse']
    area = bin_cv['cdf_area'] <= bin_cv['gt_area']

    bin_cv['domain_rmse/std_y'] = np.where(rmse, 'ID', 'OD')
    bin_cv['domain_cdf_area'] = np.where(area, 'ID', 'OD')

    return data_cv, bin_cv


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
                 precs=[],
                 gt_rmse=None,
                 gt_area=None,
                 disable_tqdm=False,
                 ):

        '''
        inputs:
            gs_model = The grid search enemble model.
            ds_model = The distance model.
            uq_model = The UQ model
            splits = The list of splitting generators.
            bins = The number of quantailes for binning data.
            precs = The minimum preicisions for domain model.
            gt_rmse = The ground truth for rmse.
            gt_area = The ground truth for miscalibration area.
        '''

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model
        self.bins = bins
        self.gt_rmse = gt_rmse
        self.gt_area = gt_area
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
        data['y_stdu_pred'] = predict_std(gs_model_cv, X_trans_te)
        data['d_pred'] = ds_model_cv.predict(X_trans_te)
        data['r'] = y[te]-data['y_pred']
        data['r/std_y'] = data['r']/data['std_y']

        return data

    def fit(self, X, y, g=None, d_input=None):
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
        data_cv = data_cv[data_cv['splitter'] != 'calibration']

        # Fit models
        self.gs_model.fit(X, y)

        self.uq_model.fit(
                          data_id['y'].values,
                          data_id['y_pred'].values,
                          data_id['y_stdu_pred'].values
                          )

        X_trans = pipe_transforms(
                                  self.gs_model,
                                  X,
                                  )

        self.ds_model.fit(X_trans)

        # Update data not used for calibration of UQ
        data_cv['y_stdc_pred'] = self.uq_model.predict(data_cv['y_stdu_pred'])
        data_cv['y_stdc_pred/std_y'] = data_cv['y_stdc_pred']/data_cv['std_y']
        data_cv['z'] = data_cv['r']/data_cv['y_stdc_pred']

        # Get binned data from alternate forms of sampling
        data_cv, bin_cv = bin_data(data_cv, self.bins)

        # Acquire ground truths
        self = ground_truth(self, y)
        data_cv['gt_rmse'] = self.gt_rmse
        data_cv['gt_area'] = self.gt_area

        # Classify ground truth labels
        data_cv, bin_cv = assign_ground_truth(
                                              data_cv,
                                              bin_cv,
                                              )

        # Fit domain classifiers
        self.domain_rmse = domain(self.precs)
        self.domain_area = domain(self.precs)

        # Train classifiers
        self.domain_rmse.fit(
                             data_cv['d_pred_max'].values,
                             data_cv['domain_rmse/std_y'].values,
                             )
        self.domain_area.fit(
                             data_cv['d_pred_max'].values,
                             data_cv['domain_cdf_area'].values,
                             )

        pred = self.combine_domains_preds(data_cv['d_pred'], d_input)
        data_cv = pd.concat([
                             data_cv.reset_index(drop=True),
                             pred.reset_index(drop=True),
                             ], axis=1)

        self.data_cv = data_cv
        self.bin_cv = bin_cv

    def combine_domains_preds(self, d, d_input=None):
        '''
        Combine domain classifiers that were fit for RMSE
        and miscalibration area.
        '''

        # Predict domains on training data
        data_rmse_dom_pred = self.domain_rmse.predict(d, d_input)
        data_rmse_dom_pred = data_rmse_dom_pred.add_prefix('rmse/std_y ')

        data_area_dom_pred = self.domain_area.predict(d, d_input)
        data_area_dom_pred = data_area_dom_pred.add_prefix('cdf_area ')

        dom_pred = pd.concat([
                              data_rmse_dom_pred,
                              data_area_dom_pred,
                              ], axis=1)

        return dom_pred

    def predict(self, X, d_input=None):
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
                          self.combine_domains_preds(pred['d_pred'], d_input),
                          ], axis=1)

        return pred

    def plot(self, save):
        '''
        Plot model fit data.
        '''

        plot = plotter(
                       self.data_cv,
                       self.bin_cv,
                       self.precs,
                       save,
                       )
        plot.generate()
