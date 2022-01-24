from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mad.datasets import load_data, aggregate, statistics
from mad.plots import parity, calibration, err_in_err, nllh, nllh_versus
from mad.ml import splitters, domain, feature_selectors
from mad.functions import poly

import numpy as np
import unittest
import shutil


class ml_test(unittest.TestCase):

    def test_ml(self):
        '''
        Test random forst model.
        '''

        seed = 14987
        save = 'run'
        uq_func = poly
        uq_coeffs_start = [0.0, 1.1, 0.0]

        # Load data
        data = load_data.sigmoid(3)
        df = data['frame']
        X = data['data']
        y = data['target']
        d = data['class_name']

        # Splitters
        top_split = splitters.BootstrappedLeaveOneGroupOut(1, d)
        mid_split = RepeatedKFold(n_splits=5, n_repeats=1)
        bot_split = RepeatedKFold(n_splits=5, n_repeats=1)

        # ML setup
        scale = StandardScaler()
        selector = feature_selectors.no_selection()

        # Random forest regression
        grid = {}
        model = RandomForestRegressor()
        grid['model__n_estimators'] = [100]
        grid['model__max_features'] = [None]
        grid['model__max_depth'] = [None]
        pipe = Pipeline(steps=[
                               ('scaler', scale),
                               ('select', selector),
                               ('model', model)
                               ])
        rf = GridSearchCV(pipe, grid, cv=bot_split)

        # Evaluate
        splits = domain.builder(
                                rf,
                                X,
                                y,
                                d,
                                top_split,
                                mid_split,
                                save,
                                seed=seed,
                                uq_func=poly,
                                uq_coeffs_start=uq_coeffs_start
                                )
        splits.assess_domain()  # Do ML
        splits.aggregate()
        statistics.folds(save)  # Gather statistics from data
        parity.make_plots(save, 'pdf')  # Make parity plots
        calibration.make_plots(save, 10, 'stdcal', 'pdf')
        err_in_err.make_plots(save, 10, 'mahalanobis')
        nllh_versus.make_plots(save, 10, 'mahalanobis')
        nllh.make_plots(save)

        # Clean directory
        shutil.rmtree(save)


if __name__ == '__main__':
    unittest.main()
