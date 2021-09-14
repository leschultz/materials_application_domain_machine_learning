from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn import cluster

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.datasets import load_data, aggregate, statistics
from mad.plots import rmse_versus, loglikelihood_versus
from mad.plots import kde, parity, calibration
from mad.plots import bar
from mad.ml import splitters, predict

import numpy as np
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test ml workflow
        '''

        seed = 14987
        save = './test'
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
        selector = SelectFromModel(Lasso(), threshold=-np.inf, max_features=2)
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        gpr = GridSearchCV(pipe, grid, cv=inner_split)

        # Random forest regression
        model = RandomForestRegressor()
        grid = {}
        grid['model__n_estimators'] = [100]
        grid['model__max_features'] = [None]
        grid['model__max_depth'] = [None]
        selector = SelectFromModel(Lasso(), threshold=-np.inf, max_features=2)
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        rf = GridSearchCV(pipe, grid, cv=inner_split)

        # Do LASSO
        model = Lasso()
        grid = {}
        grid['model__alpha'] = np.logspace(-2, 2, 5)
        selector = SelectFromModel(Lasso(), threshold=-np.inf, max_features=2)
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        lasso = GridSearchCV(pipe, grid, cv=inner_split)

        # Make pipeline
        pipes = [gpr, rf, lasso]

        # Evaluate
        predict.run(X, y, outer_split, pipes, save, seed)  # Perform ML
        aggregate.folds(save)  # Combine split data from directory recursively
        statistics.folds(save)  # Gather statistics from data
        bar.make_plots(save)  # Make logpdf plot for outlier cutoff
        statistics.folds(save, low_flag=-65)  # Gather statistics from data
        parity.make_plots(save)  # Make parity plots
        rmse_versus.make_plots(save, points, sampling)  # RMSE vs metrics
        loglikelihood_versus.make_plots(save, points, sampling)  # likelihood
        calibration.make_plots(save, points, sampling)  # Global calibration
        kde.make_plots(df, save)  # Global KDE plots

        # Clean directory
        shutil.rmtree(save)


if __name__ == '__main__':
    unittest.main()
