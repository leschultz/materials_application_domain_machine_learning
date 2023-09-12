from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class weighted_model:

    def __init__(self, bandwidth, weights, kernel):
        self.bandwidths = bandwidth*weights
        self.kernel = kernel

    def fit(self, X_train):
        self.models = []
        for b in range(self.bandwidths.shape[0]):
            self.model = KernelDensity(
                                       kernel=self.kernel,
                                       bandwidth=self.bandwidths[b],
                                       ).fit(X_train[:, b:b+1])

            self.models.append(self.model)

    def score_samples(self, X):
        scores = []
        for b in range(self.bandwidths.shape[0]):
            score = self.models[b].score_samples(X[:, b:b+1])
            scores.append(score)

        return np.sum(scores, axis=0)

    def return_bandwidths(self):
        return self.bandwidths


class distance_model:

    def __init__(self, dist='kde', weights=None, *args, **kwargs):

        self.dist = dist
        self.weights = weights
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
            if (self.weights is None) and (self.bandwidth == 0.0):
                self.model = KernelDensity(
                                           kernel=self.kernel,
                                           ).fit(X_train)
                self.bandwidth = self.model.bandwidth  # Update

            elif (self.weights is None) and (self.bandwidth > 0.0):
                self.model = KernelDensity(
                                           kernel=self.kernel,
                                           bandwidth=self.bandwidth,
                                           ).fit(X_train)
            else:

                self.model = weighted_model(
                                            self.bandwidth,
                                            self.weights,
                                            self.kernel
                                            )
                self.model.fit(X_train)
                self.bandwidth = self.model.bandwidths

            dist = self.model.score_samples(X_train)
            m = max(dist)
            cut = 0.0  # No likelihood should be greater than that trained on
            self.scaler = lambda x: np.maximum(cut, 1-np.exp(x-m))

        else:
            self.model = lambda X_test: cdist(X_train, X_test, self.dist)

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
            dist = self.model(X)
            dist = np.mean(dist, axis=0)

        return dist
