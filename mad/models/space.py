from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

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

        self.scaler = MinMaxScaler()
        if self.dist == 'gpr':
            self.model = GaussianProcessRegressor()
            self.model.fit(X_train, y_train)
            _, dist = self.model.predict(X_train, return_std=True)

        elif self.dist == 'kde':
            bw = estimate_bandwidth(X_train)
            model = KernelDensity(bandwidth=bw).fit(X_train)

            self.model = model
            self.bw = bw

            dist = -self.model.score_samples(X_train)

        else:
            self.model = lambda X_test: cdist(X_train, X_test, self.dist)
            dist = self.model(X_train, self.dist)
            dist = np.mean(dist, axis=0)

        self.scaler.fit(dist.reshape(-1, 1))

    def predict(self, X):

        if self.dist == 'gpr':
            _, dist = self.model.predict(X, return_std=True)
        elif self.dist == 'kde':
            dist = -self.model.score_samples(X)
        else:
            dist = self.model(X, self.dist)
            dist = np.mean(dist, axis=0)

        dist = self.scaler.transform(dist.reshape(-1, 1))
        dist = dist[:, 0]

        return dist
