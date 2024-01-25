from madml.ml.selectors import ShapFeatureSelector
from sklearn.model_selection import RepeatedKFold
from madml.plots import binned_truth
from madml.utils import parallel
from sklearn.base import clone
from sklearn import metrics
from scipy import stats
from madml import plots

import pandas as pd
import numpy as np

import json
import copy
import os


def domain_pred(dist, dist_cut, domain):
    '''
    Predict the domain based on thresholds.

    inputs:
        dist = The score.
        dist_cut = The threshold.
        domain = In domain (True) or out of domain (False) class label.

    outputs:
        do_pred = The domain prediction.
    '''

    do_pred = []
    for i in dist:
        if domain is True:
            if i < dist_cut:
                do_pred.append(True)
            else:
                do_pred.append(False)
        elif domain is False:
            if i >= dist_cut:
                do_pred.append(True)
            else:
                do_pred.append(False)

    return do_pred


def domain_preds(
                 pred,
                 dists=None,
                 methods=None,
                 thresholds=None,
                 suffix='',
                 manual_th=None,
                 ):
    '''
    The domain predictor based on thresholds.

    inputs:
        pred = Previous data for predictions.
        dists = The dissimilarity measures used.
        methods = Whether data are binned and use single predictions.
        thresholds = The thresholds from PR curve.
        suffix = Whether a column needs a suffix.

    outputs:
        pred = The predictions of domain.
    '''

    if manual_th is not None:
        for i in manual_th:

            dist, domain_type, th = i

            if 'id' in domain_type:
                domain = True
            elif 'od' in domain_type:
                domain = False

            name = '{} by {} for {}'.format(domain_type.upper(), dist, th)
            do_pred = domain_pred(
                                  pred[dist],
                                  th,
                                  domain,
                                  )

            pred[name] = do_pred
    else:
        for i in dists:

            if suffix:
                col = i+suffix
            else:
                col = i

            for j, k in zip([True, False], ['id', 'od']):
                for method in methods:
                    k += method
                    for key, value in thresholds[i][k].items():

                        name = '{} by {} for {}'.format(k.upper(), i, key)

                        do_pred = domain_pred(
                                              pred[col],
                                              value['Threshold'],
                                              j,
                                              )

                        pred[name] = do_pred

    return pred


class domain_model:
    '''
    Combine distance, UQ, and ensemble regression models.
    '''

    def __init__(
                 self,
                 gs_model,
                 ds_model=None,
                 uq_model=None,
                 splits=[('fit', RepeatedKFold(n_repeats=2))],
                 bins=10,
                 save=False,
                 gts=1.0,
                 gtb=0.25,
                 ):

        '''
        inputs:
            gs_model = The grid search enemble model.
            ds_model = The distance model.
            uq_model = The UQ model
            splits = The list of splitting generators.
            bins = The number of quantailes for binning data.
            save = The location to save figures and data.
            gts = The ground truth cutoff for residual magnitude test.
            gtb = The ground truth cutoff for statistical test.
        '''

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model
        self.bins = bins
        self.save = save
        self.splits = copy.deepcopy(splits)
        self.gts = gts
        self.gtb = gtb

        self.dists = []
        self.methods = ['']
        if self.uq_model:
            self.dists.append('y_stdc/std(y)')
            self.methods.append('_bin')

            # Add a splitter to calibrate UQ and prevent overfitting
            uqsplits = []
            for i in self.splits:
                if 'fit' == i[0]:
                    i = copy.deepcopy(i[1])
                    i = ('calibration', i)
                    uqsplits.append(i)

            self.splits += uqsplits

        if self.ds_model:
            self.dists.append('dist')

    def transforms(self, gs_model, X):
        '''
        Apply all steps from a pipeline before the final prediction.

        inputs:
            gs_model = The gridsearch pipeline model.
            X = The features.
        ouputs:
            X = The features with applied transformations.
        '''

        for step in list(gs_model.best_estimator_.named_steps)[:-1]:

            step = gs_model.best_estimator_.named_steps[step]
            X = step.transform(X)

        return X

    def std_pred(self, gs_model, X_test):
        '''
        Get predictions from ensemble model.

        inputs:
            gs_model = The gridsearch pipeline model.
            X_test = The featues.

        outputs:
            std = The standard deviation between models from ensemble model.
        '''

        estimators = gs_model.best_estimator_
        estimators = estimators.named_steps['model']
        estimators = estimators.estimators_

        std = []
        for i in estimators:
            std.append(i.predict(X_test))

        std = np.std(std, axis=0)

        return std

    def cv(self, split, gs_model, ds_model, X, y, g):
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

        gs_model_cv = clone(gs_model)
        gs_model_cv.fit(X[tr], y[tr])

        data = pd.DataFrame()
        data['g'] = g[te]
        data['y'] = y[te]
        data['y_pred'] = gs_model_cv.predict(X[te])
        data['index'] = te
        data['fold'] = [count]*te.shape[0]
        data['std(y)'] = [np.std(y[tr])]*te.shape[0]  # Of the data trained on
        data['splitter'] = splitter

        if self.uq_model or self.ds_model:
            X_trans_tr = self.transforms(
                                         gs_model_cv,
                                         X[tr],
                                         )
            X_trans_te = self.transforms(
                                         gs_model_cv,
                                         X[te],
                                         )

        if self.uq_model:
            data['y_stdu'] = self.std_pred(gs_model_cv, X_trans_te)

        if self.ds_model:

            ds_model_cv = copy.deepcopy(ds_model)
            ds_model_cv.fit(X_trans_tr)

            data['dist'] = ds_model_cv.predict(X_trans_te)

        data['index'] = data['index'].astype(int)

        return data

    def fit(self, X, y, g):
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

        data_cv = parallel(
                           self.cv,
                           splits,
                           gs_model=self.gs_model,
                           ds_model=self.ds_model,
                           X=X,
                           y=y,
                           g=g,
                           )

        data_cv = pd.concat(data_cv)

        # Get some data statistics
        self.ystd = np.std(y)

        # Residuals
        data_cv['r'] = data_cv['y']-data_cv['y_pred']
        data_cv['r/std(y)'] = data_cv['r']/data_cv['std(y)']
        data_cv['y_pred/std(y)'] = data_cv['y_pred']/data_cv['std(y)']
        data_cv['y/std(y)'] = data_cv['y']/data_cv['std(y)']
        data_cv['id'] = abs(data_cv['r/std(y)']) < self.gts  # Ground truth

        # Fit UQ on hold out data ID
        if self.uq_model:

            data_id = data_cv[data_cv['splitter'] == 'calibration']
            data_cv = data_cv[data_cv['splitter'] != 'calibration']

            self.uq_model.fit(
                              data_id['y'].values,
                              data_id['y_pred'].values,
                              data_id['y_stdu'].values
                              )

            data_cv['y_stdc'] = self.uq_model.predict(data_cv['y_stdu'].values)
            data_cv['y_stdc/std(y)'] = data_cv['y_stdc']/data_cv['std(y)']
            data_cv['z'] = data_cv['r']/data_cv['y_stdc']

        # Build the model.
        self.gs_model.fit(X, y)

        if self.ds_model:

            X_trans = self.transforms(
                                      self.gs_model,
                                      X,
                                      )

            self.ds_model.fit(X_trans)

        out = plots.generate_plots(
                                   data_cv,
                                   self.ystd,
                                   self.bins,
                                   self.save,
                                   self.gts,
                                   self.gtb,
                                   self.dists,
                                   )
        self.thresholds, data_cv_bin = out

        if self.save:
            data_cv.to_csv(os.path.join(
                                        self.save,
                                        'single.csv'
                                        ), index=False)
            jsonfile = os.path.join(
                                    self.save,
                                    'thresholds.json'
                                    )
            with open(jsonfile, 'w') as handle:
                json.dump(self.thresholds, handle)

        data_cv = domain_preds(
                               data_cv,
                               self.dists,
                               self.methods,
                               self.thresholds,
                               )

        data = {}
        if 'y_stdc/std(y)' in self.dists:
            for i in self.dists:
                data[i] = domain_preds(
                                       data_cv_bin[i],
                                       [i],
                                       ['_bin'],
                                       self.thresholds,
                                       '_max',
                                       )

        data_cv_bin = data

        if self.save:
            plots.generate_confusion(data_cv, data_cv_bin, self.save)

        return data_cv, data_cv_bin

    def predict(self, X, manual_th=None):
        '''
        Give domain classification along with other regression predictions.

        inputs:
            X = The features.
            manual_th = Manual thresholds.

        outputs:
           pred = A pandas dataframe containing prediction data.
        '''

        X_trans = self.transforms(
                                  self.gs_model,
                                  X,
                                  )

        # Model predictions
        y_pred = self.gs_model.predict(X)
        y_pred_norm = y_pred/self.ystd

        pred = {
                'y_pred': y_pred,
                'y_pred/std(y)': y_pred_norm,
                }

        if self.ds_model:
            dist = self.ds_model.predict(X_trans)
            pred['dist'] = dist

        if self.uq_model:
            y_stdu = self.std_pred(self.gs_model, X_trans)
            y_stdc = self.uq_model.predict(y_stdu)
            y_stdc_norm = y_stdc/self.ystd
            y_stdu_norm = y_stdu/self.ystd

            pred['y_stdu'] = y_stdu
            pred['y_stdu/std(y)'] = y_stdu_norm
            pred['y_stdc'] = y_stdc
            pred['y_stdc/std(y)'] = y_stdc_norm

        pred = pd.DataFrame(pred)

        if manual_th is None:
            pred = domain_preds(
                                pred,
                                self.dists,
                                self.methods,
                                self.thresholds,
                                )
        else:

            pred = domain_preds(
                                pred,
                                manual_th=manual_th
                                )

        return pred
