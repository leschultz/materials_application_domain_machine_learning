from sklearn.model_selection import RepeatedKFold
from mad.functions import parallel
from sklearn.base import clone

import pandas as pd
import numpy as np

import random
import os


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

    def __init__(self, X, y, g=None, splitter=RepeatedKFold(), seed=64064):

        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.X = X  # Features
        self.y = y  # Target
        self.splitter = splitter  # Splitter

        # Grouping
        if g is None:
            self.g = np.ones(self.X.shape[0])
        else:
            self.g = g

    def split(self, X, y, g, splitter):

        # Train, test splits
        count = -1
        for split in splitter.split(X, y, g):
            train, test = split
            count += 1
            yield (train, test, count)

    def transforms(self, X_train, y_train, X_test, gs_model):

        for step in list(gs_model.best_estimator_.named_steps)[:-1]:

            step = gs_model.best_estimator_.named_steps[step]
            X_train = step.transform(X_train)
            X_test = step.transform(X_test)

        return X_train, X_test

    def fit(self, split, gs_model, uq_model, ds_model):

        train, test, count = split  # train/test

        # Fit grid search
        gs_model.fit(self.X[train], self.y[train])

        # Inner fold
        y_cv = []
        y_cv_pred = []
        y_cv_std = []
        for tr, te in gs_model.cv.split(
                                        self.X[train],
                                        self.y[train],
                                        self.g[train],
                                        ):

            cv_model = clone(gs_model)
            cv_model.fit(self.X[train][tr], self.y[train][tr])

            std = []
            for i in cv_model.best_estimator_.named_steps['model'].estimators_:
                X_train, X_test = self.transforms(
                                                  self.X[train][tr],
                                                  self.y[train][tr],
                                                  self.X[train][te],
                                                  cv_model,
                                                  )
                std.append(i.predict(X_test))

            y_cv_pred = np.append(
                                  y_cv_pred,
                                  cv_model.predict(self.X[train][te])
                                  )

            y_cv_std = np.append(
                                 y_cv_std,
                                 np.std(std, axis=0)
                                 )
            y_cv = np.append(
                             y_cv,
                             self.y[train][te]
                             )

        # Model fitting
        ds_model.fit(self.X[train], self.y[train])
        uq_model.fit(y_cv, y_cv_pred, y_cv_std)

        # Model predictions
        y_std = []
        for i in gs_model.best_estimator_.named_steps['model'].estimators_:
            X_train, X_test = self.transforms(
                                              self.X[train],
                                              self.y[train],
                                              self.X[test],
                                              gs_model,
                                              )
            y_std.append(i.predict(X_test))
        y_std = np.std(y_std, axis=0)

        y_pred = gs_model.predict(self.X[test])
        dist = ds_model.predict(self.X[test])
        y_std = uq_model.predict(y_std)

        data = {}
        data['y'] = self.y[test]
        data['y_pred'] = y_pred
        data['y_std'] = y_std
        data['dist'] = dist
        data['index'] = test

        data = pd.DataFrame(data)
        data['fold'] = count

        return data

    def predict(self, gs_model, uq_model, ds_model):

        splits = self.split(
                            self.X,
                            self.y,
                            self.g,
                            self.splitter
                            )
        splits = list(splits)

        data = parallel(
                        self.fit,
                        splits,
                        gs_model=gs_model,
                        uq_model=uq_model,
                        ds_model=ds_model,
                        )

        data = pd.concat(data)
        print(data)
