from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class distance_model:

    def __init__(self, dist='kde'):
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
            y_train = The training target when applicable.
            X_test = The features of the test set.
            dist = The distance to consider.

        ouputs:
            dists = A dictionary of distances.
        '''

        if self.dist == 'gpr':
            self.model = GaussianProcessRegressor()
            self.model.fit(X_train, y_train)
            _, dist = self.model.predict(X_train, return_std=True)
            self.scaler = lambda x: -x

        elif self.dist == 'kde':
            bw = estimate_bandwidth(X_train)

            if bw > 0.0:
                self.model = KernelDensity(bandwidth=bw).fit(X_train)
            else:
                self.model = KernelDensity().fit(X_train)

            dist = self.model.score_samples(X_train)
            self.scaler = lambda x: 1-np.exp(x-max(dist))

        else:
            self.model = lambda X_test: cdist(X_train, X_test, self.dist)
            dist = self.model(X_train, self.dist)
            dist = np.mean(dist, axis=0)

    def predict(self, X):
        '''
        Get the distances for individual cases.

        inputs:
            X = The features of data.
        outputs:
            dist = The distance array.
        '''

        if self.dist == 'gpr':
            _, dist = self.model.predict(X, return_std=True)
            dist = self.scaler(dist)

        elif self.dist == 'kde':
            dist = self.model.score_samples(X)
            dist = self.scaler(dist)

        else:
            dist = self.model(X, self.dist)
            dist = np.mean(dist, axis=0)

        return dist
