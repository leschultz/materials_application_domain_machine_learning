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
                indx = score != -np.inf
                score[indx] = score[indx]*self.weights[b]

            scores.append(score)

        return np.sum(scores, axis=0)

    def return_bandwidths(self):
        return self.bandwidths


class distance_model:

    def __init__(
                 self,
                 dist='kde',
                 kernel='epanechnikov',
                 bandwidth=None,
                 weights=None,
                 ):

        self.dist = dist
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.weights = weights

    def fit(
            self,
            X_train,
            ):
        '''
        Get the distances based on a metric.

        inputs:
            X_train = The features of the training set.
            dist = The distance to consider.
            model = The model to get feature importances.
            n_features = The number of features to keep.

        ouputs:
            dists = A dictionary of distances.
        '''

        if self.dist == 'kde':

            if self.bandwidth is None:
                self.bandwidth = estimate_bandwidth(X_train)

            if self.bandwidth > 0.0:

                if self.weights is not None:
                    model = weighted_kde(
                                         self.kernel,
                                         self.bandwidth,
                                         weigh='scores',
                                         weights=self.weights,
                                         )
                else:

                    model = KernelDensity(
                                          kernel=self.kernel,
                                          bandwidth=self.bandwidth,
                                          )

                model.fit(X_train)

                dist = model.score_samples(X_train)
                m = np.max(dist)

                def pred(X):
                    out = model.score_samples(X)
                    out = out-m
                    out = np.exp(out)
                    out = 1-out
                    out = np.maximum(0.0, out)
                    return out

                self.model = pred

            else:
                self.model = lambda x: np.repeat(1.0, len(x))

        else:

            def pred(X):
                out = cdist(X_train, X, self.dist)
                out = np.mean(out, axis=0)
                return out

            self.model = pred

    def predict(self, X):
        '''
        Get the distances for individual cases.

        inputs:
            X = The features of data.
        outputs:
            dist = The distance array.
        '''

        return self.model(X)
