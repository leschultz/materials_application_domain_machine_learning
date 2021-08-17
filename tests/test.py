from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cluster

from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.datasets import load_data
from mad.plots import kde, bins
from mad.ml import predict
from mad.ml import splitters

import numpy as np
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_loads(self):
        '''
        Test data import
        '''

        load_data.friedman()
        load_data.diffusion()
        load_data.super_cond()

    def test_ml(self):
        '''
        Test ml workflow
        '''

        save = './test'
        points = 15
        sampling = 'even'

        data = load_data.diffusion()
        df = data['frame']
        X = data['data']
        y = data['target']

        # ML setup
        scale = StandardScaler()
        inner_split = splitters.split.repcf(cluster.OPTICS, 5, 2)
        outer_split = splitters.split.repcf(cluster.OPTICS, 5, 1)

        # inner_split = splitters.repkf(5, 2)
        # outer_split = splitters.repkf(5, 10)

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
        predict.run(X, y, outer_split, pipes, save, 14987)
        bins.make_plots(save, points, sampling)
        kde.make_plots(df, save)

        # Clean directory
        shutil.rmtree(save)


if __name__ == '__main__':
    unittest.main()
