from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class weighted_model:

    def __init__(self, kernel, bandwidth=None, weights=None):

        self.bandwidth = bandwidth
        self.weights = weights
        self.kernel = kernel

    def fit(self, X_train):

        self.counts = range(X_train.shape[1])
        self.models = []
        self.bandwidths = []

        for b in self.counts:

            cut = b+1

            if self.bandwidth is None:
                model = KernelDensity(
                                      kernel=self.kernel,
                                      ).fit(X_train[:, b:cut])
                bandwidth = model.bandwidth

            else:
                bandwidth = self.bandwidth

                model = KernelDensity(
                                      kernel=self.kernel,
                                      bandwidth=bandwidth,
                                      ).fit(X_train[:, b:cut])

            self.models.append(model)
            self.bandwidths.append(bandwidth)

    def score_samples(self, X):

        scores = []
        for b in self.counts:
            score = self.models[b].score_samples(X[:, b:b+1])

            if self.weights is not None:
                score = score*self.weights[b]

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
                self.bandwidth = None

            self.model = weighted_model(
                                        self.kernel,
                                        weights=self.weights,
                                        bandwidth=self.bandwidth,
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
