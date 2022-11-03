from sklearn.model_selection import RepeatedKFold
from mad.stats.group import stats, group_metrics
from mad.utils import parallel
from mad.plots import parity, cdf_parity
from sklearn.base import clone

import pandas as pd
import numpy as np

import copy
import dill
import os


def transforms(gs_model, X):

    for step in list(gs_model.best_estimator_.named_steps)[:-1]:

        step = gs_model.best_estimator_.named_steps[step]
        X = step.transform(X)
        return X


def std_pred(gs_model, X_test):
    std = []
    estimators = gs_model.best_estimator_
    estimators = estimators.named_steps['model']
    estimators = estimators.estimators_
    X_test = transforms(
                        gs_model,
                        X_test,
                        )
    for i in estimators:
        std.append(i.predict(X_test))

    std = np.std(std, axis=0)
    return std


def cv(gs_model, ds_model, X, y, g, train):
    '''
    Do cross validation.
    '''

    y_cv = []
    y_cv_pred = []
    y_cv_std = []
    index_cv = []
    dist_cv = []
    for tr, te in gs_model.cv.split(
                                    X[train],
                                    y[train],
                                    g[train],
                                    ):

        gs_model_cv = clone(gs_model)
        ds_model_cv = copy.deepcopy(ds_model)

        gs_model_cv.fit(X[train][tr], y[train][tr])
        ds_model_cv.fit(X[train][tr], y[train][tr])

        std = std_pred(gs_model, X[train][te])

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

        index_cv = np.append(index_cv, train[te])
        dist_cv = np.append(
                            dist_cv,
                            ds_model_cv.predict(X[train][te])
                            )

    data = pd.DataFrame()
    data['y'] = y_cv
    data['y_pred'] = y_cv_pred
    data['y_std'] = y_cv_std
    data['dist'] = dist_cv
    data['index'] = index_cv
    data['split'] = 'cv'

    return data


class build_model:

    def __init__(self, gs_model, ds_model, uq_model):
        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model

    def fit(self, X, y, g):

        # Build the model
        self.gs_model.fit(X, y)
        self.ds_model.fit(X, y)

        # Do cross validation in nested loop
        data_cv = cv(
                     self.gs_model,
                     self.ds_model,
                     X,
                     y,
                     g,
                     np.arange(y.shape[0])
                     )

        # Fit on hold out data
        self.uq_model.fit(
                          data_cv['y'],
                          data_cv['y_pred'],
                          data_cv['y_std']
                          )

        # Update with calibrated data
        data_cv['y_std'] = self.uq_model.predict(data_cv['y_std'])

        return data_cv

    def predict(self, X):

        # Model predictions
        y_pred = self.gs_model.predict(X)
        y_std = std_pred(self.gs_model, X)
        y_std = self.uq_model.predict(y_std)  # Calibrate hold out
        dist = self.ds_model.predict(X)

        return y_pred, y_std, dist


class NestedCV:

    '''
    A class to split data into multiple levels.

    Parameters
    ----------

    X : numpy array
        The original features to be split.

    y : numpy array
        The original target features to be split.

    g : list or numpy array, default = None
        The groups of data to be split.
    '''

    def __init__(self, X, y, g=None, splitter=RepeatedKFold()):

        self.X = X  # Features
        self.y = y  # Target
        self.splitter = splitter  # Splitter

        # Grouping
        if g is None:
            self.g = np.ones(self.X.shape[0])
        else:
            self.g = g

        # Generate the splits
        splits = self.split(
                            self.X,
                            self.y,
                            self.g,
                            self.splitter
                            )
        self.splits = list(splits)

    def split(self, X, y, g, splitter):

        # Train, test splits
        count = -1
        for split in splitter.split(X, y, g):
            train, test = split
            count += 1
            yield (train, test, count)

    def fit(self, split, gs_model, uq_model, ds_model):

        train, test, count = split  # train/test

        # Fit models
        model = build_model(gs_model, ds_model, uq_model)
        data_cv = model.fit(self.X[train], self.y[train], self.g[train])
        preds = model.predict(self.X[test])
        y_pred, y_std, dist = preds

        data_test = pd.DataFrame()
        data_test['y'] = self.y[test]
        data_test['y_pred'] = y_pred
        data_test['y_std'] = y_std
        data_test['dist'] = dist
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'

        data_cv['fold'] = count

        data = pd.concat([data_cv, data_test])
        data['index'] = data['index'].astype(int)

        return data

    def save_model(self, gs_model, uq_model, ds_model, save='.'):
        '''
        Build one model on all data.
        '''

        # Build the model
        model = build_model(gs_model, ds_model, uq_model)
        data_cv = model.fit(self.X, self.y, self.g)
        data_cv['fold'] = 0
        data_cv['split'] = 'cv'
        data_cv['index'] = data_cv['index'].astype(int)

        # Statistics
        df_stats = stats(data_cv, ['split', 'index'])
        mets = group_metrics(data_cv, ['split', 'fold'])
        mets = stats(mets, ['split'])

        # Save location
        original_loc = os.path.join(save, 'model')
        os.makedirs(original_loc, exist_ok=True)

        # Plot CDF comparison
        x = (data_cv['y']-data_cv['y_pred'])/data_cv['y_std']
        cdf_parity(x, save=os.path.join(original_loc, 'cv'))

        # Plot parity
        parity(
               mets,
               df_stats['y_mean'].values,
               df_stats['y_pred_mean'].values,
               save=os.path.join(original_loc, 'cv')
               )

        # Save the model
        dill.dump(model, open(os.path.join(original_loc, 'model.dill'), 'wb'))

        # Data
        pd.DataFrame(self.X).to_csv(os.path.join(
                                                 original_loc,
                                                 'X.csv'
                                                 ), index=False)
        pd.DataFrame(self.y).to_csv(os.path.join(
                                                 original_loc,
                                                 'y.csv'
                                                 ), index=False)
        pd.DataFrame(self.g).to_csv(os.path.join(
                                                 original_loc,
                                                 'g.csv'
                                                 ), index=False)

        data_cv.to_csv(os.path.join(
                                    original_loc,
                                    'cv.csv'
                                    ), index=False)

    def assess(self, gs_model, uq_model, ds_model, save='.'):

        print('Assessing splits with ML pipeline: {}'.format(save))
        data = parallel(
                        self.fit,
                        self.splits,
                        gs_model=gs_model,
                        uq_model=uq_model,
                        ds_model=ds_model,
                        )

        data = pd.concat(data)

        # Statistics
        df_stats = stats(data, ['split', 'index'])
        mets = group_metrics(data, ['split', 'fold'])
        mets = stats(mets, ['split'])

        # Save locations
        assessment_loc = os.path.join(save, 'assessment')
        os.makedirs(assessment_loc, exist_ok=True)

        # Plot assessment
        for i in ['cv', 'test']:
            subdata = data[data['split'] == i]
            subdf = df_stats[df_stats['split'] == i]
            submets = mets[mets['split'] == i]

            # Plot CDF comparison
            x = (subdata['y']-subdata['y_pred'])/subdata['y_std']
            cdf_parity(x, save=os.path.join(assessment_loc,  '{}'.format(i)))

            # Plot parity
            parity(
                   submets,
                   subdf['y_mean'].values,
                   subdf['y_pred_mean'].values,
                   subdf['y_pred_sem'].values,
                   save=os.path.join(assessment_loc, '{}'.format(i))
                   )

        # Save csv
        data.to_csv(os.path.join(
                                 assessment_loc,
                                 'assessment.csv'
                                 ), index=False)
