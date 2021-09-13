from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

import pandas as pd
import numpy as np

import random
import os

from mad.functions import parallel


def llh(std, res, x):
    '''
    Compute the log likelihood.
    '''

    total = np.log(2*np.pi)
    total += 2*np.log(x[0]*std+x[1])
    total += (res**2)/((x[0]*std+x[1])**2)
    total *= -0.5

    return total


def set_llh(std, y, y_pred, x):
    '''
    Compute the log likelihood for a dataset.
    '''

    std = std
    res = y-y_pred

    opt = minimize(lambda x: -sum(llh(std, res, x)), x, method='nelder-mead')
    a, b = opt.x

    likes = llh(std, res, opt.x)

    return a, b, likes


def distance_link(X_train, X_test, dist_type, scaler=True):
    '''
    Get the distances based on a metric.

    inputs:
        X_train = The features of the training set.
        X_test = The features of the test set.
        dist = The distance to consider.

    ouputs:
        dists = A dictionary of distances.
    '''

    if scaler:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    dists = {}
    if dist_type == 'mahalanobis':
        # Get the inverse of the covariance matrix from training
        vi = np.linalg.inv(np.cov(X_train.T))
        dist = cdist(X_train, X_test, dist_type, VI=vi)

        dists[dist_type+'_mean'] = np.mean(dist, axis=0)
        dists[dist_type+'_max'] = np.max(dist, axis=0)
        dists[dist_type+'_min'] = np.min(dist, axis=0)
    elif dist_type == 'logpdf':

        X_train = X_train.T
        X_test = X_test.T

        gaus = gaussian_kde(X_train)
        dist = gaus.logpdf(X_test)
        dists[dist_type] = dist

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
                'logpdf',
                'euclidean',
                'minkowski',
                'cityblock',
                'seuclidean',
                'sqeuclidean',
                'cosine',
                'correlation',
                'hamming',
                'jaccard',
                'jensenshannon',
                'chebyshev',
                'canberra',
                'braycurtis',
                'yule',
                'dice',
                'kulsinski',
                'rogerstanimoto',
                'russellrao',
                'sokalmichener',
                'sokalsneath',
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

    trains = []
    tests = []
    for pipe in pipes:

        train = {}
        test = {}

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
            y_train_pred = pipe_best.predict(X_train)

            X_test_trans = pipe_best_scaler.transform(X_test)
            X_train_trans = pipe_best_scaler.transform(X_train)
            pipe_estimators = pipe_best_model.estimators_

            # Ensemble predictions
            std_test = [i.predict(X_test_trans) for i in pipe_estimators]
            std_train = [i.predict(X_train_trans) for i in pipe_estimators]

            std_test = np.std(std_test, axis=0)
            std_train = np.std(std_train, axis=0)

            train['std'] = std_train
            test['std'] = std_test

        # If model is gaussian process regressor
        elif model_type == 'GaussianProcessRegressor':
            y_test_pred, std_test = pipe_best.predict(X_test, return_std=True)
            y_train_pred, std_train = pipe_best.predict(
                                                        X_train,
                                                        return_std=True
                                                        )

            train['std'] = std_train
            test['std'] = std_test

        # Add std calibrated here and include as distance metric
        if 'std' in test.keys():

            # Calibration
            a, b, likes = set_llh(std_train, y_train, y_train_pred, [0, 1])
            std_train_cal = a*std_train+b
            std_test_cal = a*std_test+b

            # Log likelihoods
            llh_vals_train = llh(std_train, y_train-y_train_pred, [a, b])
            llh_vals_test = llh(std_test, y_test-y_test_pred, [a, b])

            train['std_cal'] = std_train_cal
            train['loglikelihood'] = llh_vals_train
            test['std_cal'] = std_test_cal
            test['loglikelihood'] = llh_vals_test

        # Assign values that should be the same
        train['pipe'] = test['pipe'] = pipe
        train['model'] = test['model'] = model_type
        train['scaler'] = test['scaler'] = scaler_type
        train['splitter'] = test['spliter'] = split_type
        train['split_id'] = test['split_id'] = count

        # Training data
        train['y_train'] = y_train
        train['y_train_pred'] = y_train_pred
        train['index'] = tr_indx

        # Testing data
        test['y_test'] = y_test
        test['y_test_pred'] = y_test_pred
        test['index'] = te_indx

        test.update(dists)  # Only include distances in test

        train = pd.DataFrame(train)
        test = pd.DataFrame(test)

        trains.append(train)
        tests.append(test)

    train_name = os.path.join(
                              save,
                              'train_split_{}.csv'.format(count)
                              )

    test_name = os.path.join(
                             save,
                             'test_split_{}.csv'.format(count)
                             )
    pd.concat(trains).to_csv(train_name, index=False)
    pd.concat(tests).to_csv(test_name, index=False)


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
