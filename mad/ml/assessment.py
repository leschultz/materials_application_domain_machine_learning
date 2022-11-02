from sklearn.model_selection import RepeatedKFold
from mad.functions import parallel
from sklearn.base import clone

import pandas as pd
import numpy as np

import copy
import os

ensemble_methods = [
                    'RandomForestRegressor',
                    'BaggingRegressor',
                    ]


class build_model:

    def __init__(self, gs_model, ds_model, uq_model):
        self.gs_model = gs_model
        self.ds_model = ds_model
        self.uq_model = uq_model

    def fit(self, X, y):

        # Build the model
        self.gs_model.fit(X, y)
        self.ds_model.fit(X, y)

        # Do cross validation in nested loop
        data_cv = self.cv(
                          self.gs_model,
                          self.ds_model,
                          np.arange(y.shape[0])
                          )

        # Fit on hold out data
        sef.uq_model.fit(
                         data_cv['y'],
                         data_cv['y_pred'],
                         data_cv['y_std']
                         )

    def predict(self, X):

        # Model predictions
        y_pred = self.gs_model.predict(self.X)
        y_std = self.std_pred(gs_model, self.X)
        y_std = self.uq_model.predict(y_std)  # Calibrate hold out
        dist = ds_model.predict(self.X)

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

    def transforms(self, gs_model, X_train=None, X_test=None):

        for step in list(gs_model.best_estimator_.named_steps)[:-1]:

            step = gs_model.best_estimator_.named_steps[step]

            if X_train is not None:
                X_train = step.transform(X_train)
                return X_train
            elif X_test is not None:
                X_test = step.transform(X_test)
                return X_test
            else:
                X_train = step.transform(X_train)
                X_test = step.transform(X_test)
                return X_train, X_test

    def std_pred(self, gs_model, X_test):
        std = []
        estimators = gs_model.best_estimator_
        estimators = estimators.named_steps['model']
        estimators = estimators.estimators_
        X_test = self.transforms(
                                 gs_model,
                                 X_test=X_test,
                                 )
        for i in estimators:
            std.append(i.predict(X_test))

        std = np.std(std, axis=0)
        return std

    def cv(self, gs_model, ds_model, train):

        y_cv = []
        y_cv_pred = []
        y_cv_std = []
        index_cv = []
        dist_cv = []
        for tr, te in gs_model.cv.split(
                                        self.X[train],
                                        self.y[train],
                                        self.g[train],
                                        ):

            gs_model_cv = clone(gs_model)
            ds_model_cv = copy.deepcopy(ds_model)

            gs_model_cv.fit(self.X[train][tr], self.y[train][tr])
            ds_model_cv.fit(self.X[train][tr], self.y[train][tr])

            std = self.std_pred(gs_model, self.X[train][te])

            y_cv_pred = np.append(
                                  y_cv_pred,
                                  gs_model_cv.predict(self.X[train][te])
                                  )

            y_cv_std = np.append(
                                 y_cv_std,
                                 std
                                 )
            y_cv = np.append(
                             y_cv,
                             self.y[train][te]
                             )

            index_cv = np.append(index_cv, train[te])
            dist_cv = np.append(
                                dist_cv,
                                ds_model_cv.predict(self.X[train][te])
                                )

            data = {}
            data['y'] = y_cv
            data['y_pred'] = y_cv_pred
            data['y_std'] = y_cv_std
            data['dist'] = dist_cv
            data['index'] = index_cv

            return data

    def fit(self, split, gs_model, uq_model, ds_model):

        train, test, count = split  # train/test

        # Fit models
        gs_model.fit(self.X[train], self.y[train])
        ds_model.fit(self.X[train], self.y[train])

        # Do cross validation in nested loop
        data_cv = self.cv(
                          gs_model,
                          ds_model,
                          train
                          )

        # Fit on hold out data
        uq_model.fit(
                     data_cv['y'],
                     data_cv['y_pred'],
                     data_cv['y_std']
                     )

        # Model predictions
        y_pred = gs_model.predict(self.X[test])
        y_std = self.std_pred(gs_model, self.X[test])
        y_std = uq_model.predict(y_std)  # Calibrate hold out
        y_cv_std = uq_model.predict(data_cv['y_std'])  # Calibrate CV
        dist = ds_model.predict(self.X[test])

        data_test = pd.DataFrame()
        data_test['y'] = self.y[test]
        data_test['y_pred'] = y_pred
        data_test['y_std'] = y_std
        data_test['dist'] = dist
        data_test['index'] = test
        data_test['fold'] = count
        data_test['split'] = 'test'

        data_cv = pd.DataFrame(data_cv)
        data_cv['y_std'] = y_cv_std  # Replace with calibrated STD
        data_cv['fold'] = count
        data_cv['split'] = 'cv'

        data = pd.concat([data_cv, data_test])
        data['index'] = data['index'].astype(int)

        return data

    def save_model(self, gs_model, uq_model, ds_model, save='.'):
        '''
        Build one model on all data.
        '''

        # Build the model

        # Save data used for fitting model
        original_loc = os.path.join(save, 'model')
        os.makedirs(original_loc, exist_ok=True)
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

        # Save assessment data
        assessment_loc = os.path.join(save, 'assessment')
        os.makedirs(assessment_loc, exist_ok=True)
        data.to_csv(os.path.join(
                                 assessment_loc,
                                 'assessment.csv'
                                 ), index=False)
