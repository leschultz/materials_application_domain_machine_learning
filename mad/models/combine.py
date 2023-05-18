from sklearn.base import clone
from functools import reduce
from sklearn import metrics
from scipy import stats
from mad import plots

import pandas as pd
import numpy as np
import copy
import os


class domain_model:

    def __init__(
                 self,
                 gs_model,
                 ds_model,
                 uq_model,
                 splits,
                 bins=10,
                 save=False,
                 ):

        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model
        self.splits = splits
        self.bins = bins
        self.save = save

    def transforms(self, gs_model, X):

        for step in list(gs_model.best_estimator_.named_steps)[:-1]:

            step = gs_model.best_estimator_.named_steps[step]
            X = step.transform(X)

        return X

    def std_pred(self, gs_model, X_test):

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
        data['sigma_y'] = sigma_y  # Of the data trained on

        data['index'] = data['index'].astype(int)

        return data

    def fit(self, X, y, g):

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

        # Normalized
        data_cv['y_stdc/std(y)'] = data_cv['y_stdc']/data_cv['sigma_y']
        data_cv['y_pred/std(y)'] = data_cv['y_pred']/data_cv['sigma_y']
        data_cv['y/std(y)'] = data_cv['y']/data_cv['sigma_y']

        # Ground truth
        data_cv['id'] = data_cv['r'] < data_cv['sigma_y']

        # Get data for bins
        data_cv_bin = data_cv[[
                               'y_stdc/std(y)',
                               'dist',
                               'r',
                               'z'
                               ]].copy()

        data_cv_bin['bin'] = pd.qcut(
                                     data_cv_bin['dist'],
                                     self.bins,
                                     )

        # Bin statistics
        bin_groups = data_cv_bin.groupby('bin')
        distmean = bin_groups['dist'].mean()
        stdmean = bin_groups['y_stdc/std(y)'].mean()
        zvar = bin_groups['z'].var()
        rmse = bin_groups['r'].apply(lambda x: (sum(x**2)/len(x))**0.5)
        pvals = bin_groups['z'].apply(lambda x: stats.cramervonmises(
                                                                     x,
                                                                     'norm',
                                                                     (0, 1)
                                                                     ).pvalue)
        counts = bin_groups['r'].count()

        distmean = distmean.to_frame().add_suffix('_mean')
        stdmean = stdmean.to_frame().add_suffix('_mean')
        zvar = zvar.to_frame().add_suffix('_var')
        rmse = rmse.to_frame().rename({'r': 'rmse'}, axis=1)
        pvals = pvals.to_frame().rename({'z': 'pval'}, axis=1)
        counts = counts.to_frame().rename({'r': 'count'}, axis=1)

        data_cv_bin = reduce(
                             lambda x, y: pd.merge(x, y, on='bin'),
                             [
                              distmean,
                              stdmean,
                              zvar,
                              rmse,
                              pvals,
                              counts,
                              ]
                             )
        data_cv_bin = data_cv_bin.reset_index()
        data_cv_bin['dist_min'] = data_cv_bin['bin'].apply(lambda x: x.left).astype(float)
        data_cv_bin['dist_max'] = data_cv_bin['bin'].apply(lambda x: x.right).astype(float)

        # Ground truth for bins
        data_cv_bin['id']  = data_cv_bin['pval'] > 0.01

        self.data_cv = data_cv
        self.data_cv_bin = data_cv_bin


        prsave = os.path.join(self.save, 'pr')
        pridsave = os.path.join(prsave, 'id')
        thresh = plots.pr(
                          data_cv['dist'],
                          data_cv['id'],
                          True,
                          save=os.path.join(pridsave, 'single'),
                          )
        
        thresh_bin = plots.pr(
                              data_cv_bin['dist_max'].values,
                              data_cv_bin['id'],
                              True,
                              save=os.path.join(pridsave, 'bin')
                              )


        return data_cv, data_cv_bin

    def predict(self, X):

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

        pred = pd.DataFrame(pred)

        return pred
