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
                 ds_model,
                 uq_model,
                 splits,
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
        self.splits = splits
        self.bins = bins
        self.save = save

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

    def cv(self, gs_model, ds_model, X, y, g, train, cv):
        '''
        Do cross validation.

        inputs:
            gs_model = The grisearch pipeline model.
            ds_model = The distance model.
            X = The features.
            y = The target variable.
            g = The groups for data.
            train = The indexes for training data.
            cv = The cross validation generator for splits.

        outputs:
            data = A pandas dataframe containing CV results.
        '''

        g_cv = []
        y_cv = []
        y_cv_pred = []
        y_cv_std = []
        index_cv = []
        dist_cv = []
        sigma_y = []
        split_count = []
        for count, (tr, te) in enumerate(cv.split(
                                                  X[train],
                                                  y[train],
                                                  g[train],
                                                  )):

            if tr.shape[0] < 1:
                continue

            gs_model_cv = clone(gs_model)
            ds_model_cv = copy.deepcopy(ds_model)

            gs_model_cv.fit(X[train][tr], y[train][tr])

            X_trans_tr = self.transforms(
                                         gs_model_cv,
                                         X[train][tr],
                                         )

            X_trans_te = self.transforms(
                                         gs_model_cv,
                                         X[train][te],
                                         )

            ds_model_cv.fit(X_trans_tr, y[train][tr])

            std = self.std_pred(gs_model_cv, X_trans_te)

            y_cv_pred = np.append(
                                  y_cv_pred,
                                  gs_model_cv.predict(X[train][te])
                                  )

            y_cv_std = np.append(
                                 y_cv_std,
                                 std
                                 )
            y_cv = np.append(
                             y_cv,
                             y[train][te]
                             )
            g_cv = np.append(
                             g_cv,
                             g[train][te]
                             )

            index_cv = np.append(index_cv, train[te])
            dist_cv = np.append(
                                dist_cv,
                                ds_model_cv.predict(X_trans_te)
                                )

            cases = len(y[train][te])
            sigma_y += [np.std(y[train][tr])]*cases
            split_count += [count]*cases

        data = pd.DataFrame()
        data['g'] = g_cv
        data['y'] = y_cv
        data['y_pred'] = y_cv_pred
        data['y_stdu'] = y_cv_std
        data['dist'] = dist_cv
        data['index'] = index_cv
        data['fold'] = split_count
        data['std(y)'] = sigma_y  # Of the data trained on

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

        # Get some data statistics
        self.ystd = np.std(y)

        # Build the model
        self.gs_model.fit(X, y)

        X_trans = self.transforms(
                                  self.gs_model,
                                  X,
                                  )
        self.ds_model.fit(X_trans, y)

        # Do cross validation in nested loop
        data_cv = []
        for split in self.splits:

            data_id = self.cv(
                              self.gs_model,
                              self.ds_model,
                              X,
                              y,
                              g,
                              np.arange(y.shape[0]),
                              split[1],
                              )

            if 'calibration' == split[0]:

                # Fit on hold out data ID
                self.uq_model.fit(
                                  data_id['y'],
                                  data_id['y_pred'],
                                  data_id['y_stdu']
                                  )

            data_id['type'] = split[0]
            data_cv.append(data_id)

        data_cv = pd.concat(data_cv)

        # Calibrate uncertainties
        data_cv['y_stdc'] = self.uq_model.predict(data_cv['y_stdu'])

        # z score
        data_cv['r'] = data_cv['y']-data_cv['y_pred']
        data_cv['z'] = data_cv['r']/data_cv['y_stdc']
        data_cv['r/std(y)'] = data_cv['r']/data_cv['std(y)']

        # Normalized
        data_cv['y_stdc/std(y)'] = data_cv['y_stdc']/data_cv['std(y)']
        data_cv['y_pred/std(y)'] = data_cv['y_pred']/data_cv['std(y)']
        data_cv['y/std(y)'] = data_cv['y']/data_cv['std(y)']

        data_cv['id'] = abs(data_cv['r/std(y)']) < 1.0

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
                                        ))
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
        y_stdu = self.std_pred(self.gs_model, X_trans)
        y_stdc = self.uq_model.predict(y_stdu)  # Calibrate hold out
        y_stdc_norm = y_stdc/self.ystd
        y_stdu_norm = y_stdu/self.ystd
        dist = self.ds_model.predict(X_trans)

        pred = {
                'y_pred': y_pred,
                'y_pred/std(y)': y_pred_norm,
                'y_stdu': y_stdu,
                'y_stdu/std(y)': y_stdu_norm,
                'y_stdc': y_stdc,
                'y_stdc/std(y)': y_stdc_norm,
                'dist': dist,
                }

        for i in ['y_stdc/std(y)', 'dist']:
            for j, k in zip([True, False], ['id', 'od']):
                for key, value in self.thresholds[i][k].items():
                    thr = value['Threshold']
                    do_pred = self.domain_pred(
                                               pred[i],
                                               thr,
                                               j,
                                               )

                    if j is True:
                        pred['ID by {} for {}'.format(i, key)] = do_pred
                    else:
                        pred['OD by {} for {}'.format(i, key)] = do_pred

        pred = pd.DataFrame(pred)

        return pred
