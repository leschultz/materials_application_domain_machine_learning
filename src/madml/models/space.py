from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

import numpy as np


class distance_model:

    def __init__(
                 self,
                 dist='kde',
                 kernel='epanechnikov',
                 bandwidth=None,
                 ):

        self.dist = dist
        self.kernel = kernel
        self.bandwidth = bandwidth

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

        dist = self.model(X)

        return dist
