from sklearn.model_selection import RepeatedKFold
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


class domain_model:
    '''
    Combine distance, UQ, and ensemble regression models.
    '''

    def __init__(
                 self,
                 gs_model,
                 ds_model=None,
                 uq_model=None,
                 splits=[('calibration', RepeatedKFold(n_repeats=2))],
                 bins=10,
                 save=False,
                 ):

        '''
        inputs:
            gs_model = The grid search enemble model.
            ds_model = The distance model.
            uq_model = The UQ model
            splits = The list of splitting generators.
            bins = The number of quantailes for binning data.
            save = The location to save figures and data.
        '''

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model
        self.bins = bins
        self.save = save
        self.splits = splits

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

        if tr.shape[0] < 1:
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

    def domain_pred(self, dist, dist_cut, domain):
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
        data_cv['id'] = abs(data_cv['r/std(y)']) < 1.0  # Ground truth

        # Fit on hold out data ID
        if self.uq_model:
            data_id = data_cv[data_cv['splitter'] == 'calibration']
            self.uq_model.fit(
                              data_id['y'].values,
                              data_id['y_pred'].values,
                              data_id['y_stdu'].values
                              )
            data_cv['y_stdc'] = self.uq_model.predict(data_cv['y_stdu'].values)
            data_cv['y_stdc/std(y)'] = data_cv['y_stdc']/data_cv['std(y)']
            data_cv['z'] = data_cv['r']/data_cv['y_stdc']

        # Build the model
        self.gs_model.fit(X, y)

        if self.ds_model:
            X_trans = self.transforms(
                                      self.gs_model,
                                      X,
                                      )

            # Fit distance model
            self.ds_model.fit(X_trans)

        out = plots.generate_plots(
                                   data_cv,
                                   self.ystd,
                                   self.bins,
                                   self.save
                                   )
        th, data_cv_bin = out

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
                json.dump(th, handle)

        self.thresholds = th
        return data_cv, data_cv_bin

    def predict(self, X):
        '''
        Give domain classification along with other regression predictions.

        inputs:
            X = The features.

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

        dists = []
        methods = ['']
        if self.uq_model:
            dists.append('y_stdc/std(y)')
            methods.append('_bin')
        if self.ds_model:
            dists.append('dist')

        for i in dists:
            for j, k in zip([True, False], ['id', 'od']):
                for method in methods:
                    k += method
                    for key, value in self.thresholds[i][k].items():
                        thr = value['Threshold']
                        do_pred = self.domain_pred(
                                                   pred[i],
                                                   thr,
                                                   j,
                                                   )

                        name = '{} by {} for {}'.format(k.upper(), i, key)
                        pred[name] = do_pred

        pred = pd.DataFrame(pred)

        return pred
