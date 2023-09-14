from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class weighted_kde:

    def __init__(
                 self,
                 kernel,
                 bandwidth=None,
                 weigh=False,
                 weights=None,
                 ):

        self.bandwidth = bandwidth
        self.weights = weights
        self.weigh = weigh
        self.kernel = kernel

    def fit(self, X_train):
        '''
        Build kernel density estimate models for each feature.

        inputs:
            X_train: The training space of the model.
        '''

        self.counts = range(X_train.shape[1])
        self.models = []
        self.bandwidths = []

        for b in self.counts:

            cut = b+1

            if self.bandwidth is None:
                bandwidth = estimate_bandwidth(X_train[:, b:cut])
            else:
                bandwidth = self.bandwidth

            if self.weigh == 'bandwidth':
                bandwidth = self.weights[b]*bandwidth

            if bandwidth > 0.0:

                model = KernelDensity(
                                      kernel=self.kernel,
                                      bandwidth=bandwidth,
                                      ).fit(X_train[:, b:cut])
            else:
                model = None

            self.models.append(model)
            self.bandwidths.append(bandwidth)

    def score_samples(self, X):
        '''
        A method to use each feature KDE model to predict on new data.

        inputs:
            X = The features and cases.
        '''

        scores = []
        for b in self.counts:

            if self.models[b] is None:
                continue

            score = self.models[b].score_samples(X[:, b:b+1])

            if self.weigh == 'scores':
                score = score*self.weights[b]

            scores.append(score)

        return np.sum(scores, axis=0)

    def return_bandwidths(self):
        return self.bandwidths


class distance_model:

    def __init__(
                 self,
                 dist='kde',
                 weigh=False,
                 weights=None,
                 *args,
                 **kwargs
                 ):

        self.dist = dist
        self.weigh = weigh
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

        if self.weigh == 'features':
            X_train = X_train*self.weights

        if self.dist == 'kde':

            if 'kernel' in self.kwargs.keys():
                self.kernel = self.kwargs['kernel']
            else:
                self.kernel = 'epanechnikov'

            if 'bandwidth' in self.kwargs.keys():
                self.bandwidth = self.kwargs['bandwidth']
            else:
                self.bandwidth = None

            self.model = weighted_kde(
                                      self.kernel,
                                      weigh=self.weigh,
                                      weights=self.weights,
                                      bandwidth=self.bandwidth,
                                      )
            self.model.fit(X_train)
            self.bandwidth = self.model.bandwidths

            dist = self.model.score_samples(X_train)
            m = np.max(dist)
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

        if self.weigh == 'features':
            X = X*self.weights

        if self.dist == 'kde':
            dist = self.model.score_samples(X)
            dist = self.scaler(dist)

        else:
            dist = np.mean(self.model(X), axis=0)

        return dist
