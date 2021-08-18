from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np
import random
import os

from mad.functions import parallel


def distance_link(X_train, X_test, dist_type):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.

    ouputs:
        dists = A dictionary of distances.
    '''

    # Get the inverse of the covariance matrix from training
    dists = {}
    if dist_type == 'mahalanobis':
        vi = np.linalg.inv(np.cov(X_train.T))
        dist = cdist(X_train, X_test, dist_type, VI=vi)
    else:
        dist = cdist(X_train, X_test, dist_type)

    dists[dist_type+'_mean'] = np.mean(dist, axis=0)
    dists[dist_type+'_max'] = np.max(dist, axis=0)
    dists[dist_type+'_min'] = np.min(dist, axis=0)

    return dists


def distance(X_train, X_test):
    '''
    Determine the distance from set X_test to set X_train.
    '''

    selected = [
                'mahalanobis',
                'euclidean',
                'cityblock',
                ]

    dists = {}
    for i in selected:
        dists.update(distance_link(X_train, X_test, i))

    return dists


def inner(indx, X, y, pipes, save):
    '''
    The inner loop from nested cross validation.
    '''

    # The splitting indexes and assigned id
    indx, count = indx

    tr_indx, te_indx = indx
    X_train, X_test = X[tr_indx], X[te_indx]
    y_train, y_test = y[tr_indx], y[te_indx]

    # Calculate distances from test set cases to traning set
    dists = {}
    for key, value in distance(X_train, X_test).items():
        if key in dists:
            dists[key] += list(value)
        else:
            dists[key] = list(value)

    dfs = []
    for pipe in pipes:

        df = {}

        pipe.fit(X_train, y_train)
        pipe_best = pipe.best_estimator_

        pipe_best_model = pipe_best.named_steps['model']
        pipe_best_scaler = pipe_best.named_steps['scaler']

        model_type = pipe_best_model.__class__.__name__
        scaler_type = pipe_best_scaler.__class__.__name__
        split_type = pipe.cv.__class__.__name__

        # If model is random forest regressor
        if model_type == 'RandomForestRegressor':
            y_test_pred = pipe_best.predict(X_test)
            X_trans = pipe_best_scaler.transform(X_test)
            pipe_estimators = pipe_best_model.estimators_
            std = [i.predict(X_trans) for i in pipe_estimators]
            std = np.std(std, axis=0)
            df['std'] = std

        # If model is gaussian process regressor
        elif model_type == 'GaussianProcessRegressor':
            y_test_pred, std = pipe_best.predict(X_test, return_std=True)
            df['std'] = std

        df['pipe'] = pipe
        df['model'] = model_type
        df['scaler'] = scaler_type
        df['spliter'] = split_type
        df['y_test'] = y_test
        df['y_test_pred'] = y_test_pred
        df['index'] = te_indx
        df['split_id'] = count
        df.update(dists)

        name = os.path.join(
                            save,
                            'split_{}.csv'.format(count)
                            )

        df = pd.DataFrame(df)
        dfs.append(df)

    pd.concat(dfs).to_csv(name, index=False)


def outer(split, pipes, X, y, save):
    '''
    Save the true values, predicted values, distances, and model error.

    inputs:
        split = The splitting method.
        pipes = The machine learning pipeline.
        X = The feature matrix.
        y = The target values.
        save = The directory to save values
    '''

    # Output directory creation
    save = os.path.join(save, 'splits')
    os.makedirs(save, exist_ok=True)

    # Gather split data in parallel
    splits = list(split.split(X))
    counts = list(range(len(splits)))
    parallel(
             inner,
             list(zip(splits, counts)),
             X=X,
             y=y,
             pipes=pipes,
             save=save,
             )


def run(X, y, split, pipes, save, seed=1):
    '''
    Define the machine learning workflow with nested cross validation
    for gaussian process regression and random forest.
    '''

    # Setting seed for reproducibility
    np.random.seed(seed)
    np.random.RandomState(seed)
    random.seed(seed)

    # Nested CV
    outer(split, pipes, X, y, save)
