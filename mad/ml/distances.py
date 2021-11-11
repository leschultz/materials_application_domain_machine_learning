from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from sklearn.decomposition import *

import numpy as np


def distance_link(X_train, X_test, dist_type, append_name=''):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.
        append_name = The string to append to name of distance metric.

    ouputs:
        dists = A dictionary of distances.
    '''

    dists = {}
    if dist_type == 'mahalanobis':
        # Get the inverse of the covariance matrix from training
        if X_train.shape[1] < 2:

            vals = np.empty(X_test.shape[0])
            dists[append_name+dist_type+'_mean'] = vals
            dists[append_name+dist_type+'_max'] = vals
            dists[append_name+dist_type+'_min'] = vals

        else:
            vi = np.linalg.inv(np.cov(X_train.T))
            dist = cdist(X_train, X_test, dist_type, VI=vi)

            dists[append_name+dist_type+'_mean'] = np.mean(dist, axis=0)
            dists[append_name+dist_type+'_max'] = np.max(dist, axis=0)
            dists[append_name+dist_type+'_min'] = np.min(dist, axis=0)

    elif dist_type == 'pdf':

        '''
        # Estimate bandwidth
        grid = {
                'kernel': [
                           'gaussian',
                           'tophat',
                           'epanechnikov',
                           'exponential',
                           'linear',
                           'cosine'
                           ],
                'bandwidth': np.logspace(0, 5)
                }
        model = GridSearchCV(
                             KernelDensity(),
                             grid,
                             cv=5,
                             )
        '''

        bw = estimate_bandwidth(X_train)
        model = KernelDensity(kernel='gaussian', bandwidth=bw)
        model.fit(X_train)

        log_dist = model.score_samples(X_test)

        dists[append_name+dist_type] = np.exp(log_dist)
        dists[append_name+'log'+dist_type] = log_dist

    else:
        dist = cdist(X_train, X_test, dist_type)

        dists[append_name+dist_type+'_mean'] = np.mean(dist, axis=0)
        dists[append_name+dist_type+'_max'] = np.max(dist, axis=0)
        dists[append_name+dist_type+'_min'] = np.min(dist, axis=0)

    return dists


def distance(X_train, X_test):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    distance_list = [
                     'pdf',
                     'mahalanobis',
                     'euclidean',
                     'minkowski',
                     'cityblock',
                     'seuclidean',
                     'sqeuclidean',
                     'cosine',
                     'correlation',
                     'chebyshev',
                     'canberra',
                     'braycurtis',
                     'sokalsneath',
                     ]

    matrix_decomp_methods = [
                             PCA(),
                             SparsePCA(),
                             KernelPCA(n_components=X_train.shape[1]),
                             SparsePCA(),
                             IncrementalPCA(),
                             MiniBatchSparsePCA(),
                             ]


    # For development
    distance_list = ['pdf', 'mahalanobis', 'euclidean']
    matrix_decomp_methods = []

    dists = {}
    for distance in distance_list:

        # Compute regular distances
        dists.update(distance_link(X_train, X_test, distance))

        # Compute transformed distances
        for cur_method in matrix_decomp_methods:
            cur_method.fit(X_train)  # Refit object
            X_test_transformed = cur_method.transform(X_test)
            name = cur_method.__class__.__name__+'_'
            dists.update(distance_link(
                                       X_train,
                                       X_test_transformed,
                                       distance,
                                       name
                                       ))

    return dists
