from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cluster

from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.plots import kde, parity, calibration
from mad.plots import rmse_versus, cal_versus
from mad.plots import kde_ln_likelihood_counts
from mad.datasets import load_data, aggregate
from mad.ml import splitters, predict

import numpy as np


def main():
    '''
    Test ml workflow
    '''

    seed = 14987
    save = 'run'
    points = 15
    sampling = 'equal'

    # Load data
    data = load_data.test()
    df = data['frame']
    X = data['data']
    y = data['target']

    # ML setup
    scale = StandardScaler()
    inner_split = splitters.split.repcf(cluster.OPTICS, 5, 2)
    outer_split = splitters.split.repcf(cluster.OPTICS, 5, 2)

    # Gaussian process regression
    kernel = RBF()
    model = GaussianProcessRegressor()
    grid = {}
    grid['model__alpha'] = np.logspace(-2, 2, 5)
    grid['model__kernel'] = [RBF()]
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    gpr = GridSearchCV(pipe, grid, cv=inner_split)

    # Random forest regression
    model = RandomForestRegressor()
    grid = {}
    grid['model__n_estimators'] = [100]
    grid['model__max_features'] = [None]
    grid['model__max_depth'] = [None]
    pipe = Pipeline(steps=[('scaler', scale), ('model', model)])
    rf = GridSearchCV(pipe, grid, cv=inner_split)

    # Make pipeline
    pipes = [gpr, rf]

    # Evaluate
    predict.run(X, y, outer_split, pipes, save, seed)

    # Combine split data
    aggregate.folds(save)

    # Plots
    parity.make_plots(save)
    rmse_versus.make_plots(save, points, sampling)
    cal_versus.make_plots(save, points, sampling)
    calibration.make_plots(save, points, sampling)
    kde.make_plots(df, save)
    kde_ln_likelihood_counts.make_plots(save)


if __name__ == '__main__':
    main()
