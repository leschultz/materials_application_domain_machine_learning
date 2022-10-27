from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist

import numpy as np


class distance_model:

    def __init__(self, dist='gpr_std'):
        self.dist = dist

    def distance(
                 self,
                 X_train,
                 y_train,
                 X_test,
                 dist_type,
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

        if dist_type == 'gpr_std':

            model = GaussianProcessRegressor()
            model.fit(X_train, y_train)
            _, dist = model.predict(X_test, return_std=True)
            dist = dist

        else:
            dist = cdist(X_train, X_test, dist_type)
            dist = np.mean(dist, axis=0)

        return dist

    def fit(self, X_train, y_train):
        self.dist_func = lambda X_test: self.distance(
                                                      X_train,
                                                      y_train,
                                                      X_test,
                                                      self.dist,
                                                      )

    def predict(self, X):
        return self.dist_func(X)
