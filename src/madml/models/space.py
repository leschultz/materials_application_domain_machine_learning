from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class distance_model:

    def __init__(self, dist='kde', *args, **kwargs):

        self.dist = dist
        self.args = args
        self.kwargs = kwargs

    def fit(
            self,
            X_train,
            ):
        '''
        Get the distances based on a metric.

        inputs:
            X_train = The features of the training set.
            dist = The distance to consider.

        ouputs:
            dists = A dictionary of distances.
        '''

        if self.dist == 'kde':

            if 'kernel' in self.kwargs.keys():
                self.kernel = self.kwargs['kernel']
            else:
                self.kernel = 'epanechnikov'

            if 'bandwidth' in self.kwargs.keys():
                self.bandwidth = self.kwargs['bandwidth']
            else:
                self.bandwidth = estimate_bandwidth(X_train)

            # If the estimated bandwidth is zero
            if self.bandwidth > 0.0:
                self.model = KernelDensity(
                                           kernel=self.kernel,
                                           bandwidth=self.bandwidth,
                                           ).fit(X_train)
            else:
                self.model = KernelDensity(
                                           kernel=self.kernel,
                                           ).fit(X_train)
                self.bandwidth = self.model.bandwidth  # Update

            dist = self.model.score_samples(X_train)
            m = max(dist)
            cut = 0.0  # No likelihood should be greater than that trained on
            self.scaler = lambda x: np.maximum(cut, 1-np.exp(x-m))

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

        if self.dist == 'kde':
            dist = self.model.score_samples(X)
            dist = self.scaler(dist)

        else:
            dist = self.model(X, self.dist)
            dist = np.mean(dist, axis=0)

        return dist
