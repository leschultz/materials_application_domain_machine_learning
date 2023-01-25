from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist

import statsmodels.api as sm

import numpy as np


class distance_model:

    def __init__(self, dist='gpr_std'):
        self.dist = dist

    def fit(
            self,
            X_train,
            y_train,
            ):
        '''
        Get the distances based on a metric.
        inputs:
            X_train = The features of the training set.
            y_train = The training target when applicable.`
            X_test = The features of the test set.
            dist = The distance to consider.
        ouputs:
            dists = A dictionary of distances.
        '''

        self.X_train = X_train
        if self.dist == 'gpr_std':

            self.model = GaussianProcessRegressor()
            self.model.fit(X_train, y_train)

        elif self.dist == 'kde':

            var_type = 'c'*X_train.shape[1]
            self.model = sm.nonparametric.KDEMultivariate(
                                                          X_train,
                                                          var_type=var_type,
                                                          bw='normal_reference'
                                                          )

            self.bw = self.model.bw

        else:
            self.model = lambda X_test: cdist(X_train, X_test, self.dist)

    def predict(self, X):

        if self.dist == 'gpr_std':
            _, dist = self.model.predict(X, return_std=True)
        elif self.dist == 'kde':
            dist = -self.model.pdf(X)
        else:
            dist = self.model(X, self.dist)
            dist = np.mean(dist, axis=0)

        return dist
