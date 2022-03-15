from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from scipy.spatial.distance import cdist
from sklearn.decomposition import *
from sklearn.svm import OneClassSVM

import numpy as np


def distance_link(
                  X_train,
                  X_test,
                  dist_type,
                  append_name='',
                  y_train=None,
                  y_test=None
                  ):
    '''
    Get the distances based on a metric.
    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        append_name = The string to append to name of distance metric.
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
            dists[append_name+dist_type] = vals

        else:
            vi = np.linalg.inv(np.cov(X_train.T))
            dist = cdist(X_train, X_test, dist_type, VI=vi)

            dists[append_name+dist_type] = np.mean(dist, axis=0)

    elif dist_type == 'cosine':

        if X_train.shape[1] < 2:
            vals = np.empty(X_test.shape[0])
            dists[append_name+dist_type] = vals
        else:
            dist = cdist(X_train, X_test, metric='cosine')
            dists[append_name+dist_type] = np.mean(dist, axis=0)

    elif dist_type == 'attention_metric':

        if X_train.shape[1] < 2:
            vals = np.empty(X_test.shape[0])
            dists[append_name+dist_type] = vals
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
            dists[append_name+dist_type] = final_dist

    elif dist_type == 'pdf':

        # Estimate bandwidth and kernel
        grid = {
                'kernel': [
                           'gaussian',
                           'tophat',
                           'epanechnikov',
                           'exponential',
                           'linear',
                           'cosine'
                           ],
                'bandwidth': [estimate_bandwidth(X_train)]
                }
        model = GridSearchCV(
                             KernelDensity(),
                             grid,
                             cv=5,
                             )

        model.fit(X_train)

        log_dist = model.score_samples(X_test)
        dist = np.ma.exp(log_dist)

        dists[append_name+dist_type] = dist
        dists[append_name+'log'+dist_type] = log_dist

    elif dist_type == 'gpr_std':

        model = GaussianProcessRegressor()
        model.fit(X_train, y_train)
        _, dist = model.predict(X_test, return_std=True)
        dists[append_name+dist_type] = dist

    elif dist_type == 'oneClassSVM':
        model = OneClassSVM(gamma='auto', kernel='rbf').fit(X_train)
        log_dist = model.score_samples(X_test)
        dist = np.ma.exp(log_dist)
        dists[append_name+dist_type] = dist
        dists[append_name+'log'+dist_type] = log_dist

    elif dist_type == 'lof':
        model = LocalOutlierFactor(novelty=True).fit(X_train)
        log_dist = model.score_samples(X_test)
        dist = np.ma.exp(log_dist)
        dists[append_name+dist_type] = dist
        dists[append_name+'log'+dist_type] = log_dist

    else:
        dist = cdist(X_train, X_test, dist_type)
        dists[append_name+dist_type] = np.mean(dist, axis=0)

    return dists


def distance(X_train, X_test, y_train=None, y_test=None):
    '''
    Determine the distance from set X_test to set X_train.
    '''
    # For development
    distance_list = [
                     #'pdf',
                     #'mahalanobis',
                     #'cosine',
                     #'oneClassSVM',
                     #'attention_metric',
                     #'lof',
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
