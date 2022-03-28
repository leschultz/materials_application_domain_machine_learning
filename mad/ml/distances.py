from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist

import numpy as np


def distance_link(
                  X_train,
                  X_test,
                  dist_type,
                  y_train=None,
                  y_test=None
                  ):
    '''
    Get the distances based on a metric.
    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        y_train = The training target when applicable.
        y_test = The testing target when applicable.
    ouputs:
        dists = A dictionary of distances.
    '''

    dists = {}
    if dist_type == 'mahalanobis':
        # Get the inverse of the covariance matrix from training
        if X_train.shape[1] < 2:

            vals = np.empty(X_test.shape[0])
            dists[dist_type] = vals

        else:
            vi = np.linalg.inv(np.cov(X_train.T))
            dist = cdist(X_train, X_test, dist_type, VI=vi)

            dists[dist_type] = np.mean(dist, axis=0)

    elif dist_type == 'cosine':

        if X_train.shape[1] < 2:
            vals = np.empty(X_test.shape[0])
            dists[dist_type] = vals
        else:
            dist = cdist(X_train, X_test, metric='cosine')
            dists[dist_type] = np.mean(dist, axis=0)

    elif dist_type == 'attention_metric':

        if X_train.shape[1] < 2:
            vals = np.empty(X_test.shape[0])
            dists[dist_type] = vals
        else:
            queries = X_test
            keys = X_train
            # obtain cosine similarity range from 0 - 2 (2 means most similar)
            similarity = 2-cdist(queries, keys, metric='cosine')
            denominator = np.sum(similarity, axis=1)  # row sum

            vi = np.linalg.pinv(np.cov(keys.T))
            values = cdist(queries, keys, 'mahalanobis', VI=vi)

            final_dist = np.array(
                                  [0 for i in range(queries.shape[0])],
                                  dtype='f'
                                  )

            for i in range(len(final_dist)):
                s = np.sum(
                           [
                            (similarity[i][j]/denominator[i])*values[i][j]
                            for j in range(keys.shape[0])
                            ]
                           )
                final_dist[i] = s
            dists[dist_type] = final_dist

    elif dist_type == 'gpr_std':

        model = GaussianProcessRegressor()
        model.fit(X_train, y_train)
        _, dist = model.predict(X_test, return_std=True)
        dists[dist_type] = dist

    else:
        dist = cdist(X_train, X_test, dist_type)
        dists[dist_type] = np.mean(dist, axis=0)

    return dists


def distance(X_train, X_test, y_train=None, y_test=None):
    '''
    Determine the distance from set X_test to set X_train.
    '''
    # For development
    distance_list = [
                     'mahalanobis',
                     'cosine',
                     'attention_metric',
                     'gpr_std',
                     ]

    dists = {}
    for distance in distance_list:

        # Compute regular distances
        dists.update(distance_link(
                                   X_train,
                                   X_test,
                                   distance,
                                   y_train=y_train,
                                   y_test=y_test
                                   ))

    return dists
